package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * The matrix round trip read on the null-engine path (wala/ML#1020). A generator read made through
 * the public surface, {@link TensorGeneratorFactory#getGenerator} and {@link
 * TensorGenerator#getTensorTypes}, with no analysis run in progress has no worklist resolver
 * installed, so a read runs once and is memoized: nothing iterates and nothing settles. The
 * wala/ML#1020 change makes several walk steps answer a still-converging read with ⊥ and defer the
 * evaluating query to the resolver's settlement pass; on this path there is no such pass, so a
 * deferral that fired here would ship ⊥ ("not a tensor") as the read. Every deferral is behind the
 * finality predicate, which holds whenever no resolver is installed, so none fires here, and the
 * {@code len} fold and the value read's rank-heterogeneity gates run under the engine only, whose
 * iteration their optimism needs; this pin is the witness that the path reads as before.
 */
public class TestNullEngineRoundTrip extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_reshape_round_trip.py";

  /**
   * Builds the fixture's call graph without running the analysis and reads every {@code tf.reshape}
   * result directly through its generator, in every context.
   */
  private List<Set<TensorType>> readReshapesDirectly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles("");
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, pathFiles, FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph callGraph = builder.makeCallGraph(builder.getOptions());
    // Deliberately no performAnalysis: no resolver is installed, so the reads below take the
    // null-engine path through the public contract.
    List<Set<TensorType>> ret = new ArrayList<>();
    for (CGNode node : callGraph) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions()) {
        if (!(instruction instanceof PythonInvokeInstruction call)) continue;
        if (call.getNumberOfReturnValues() == 0) continue;
        boolean reshape = false;
        for (CGNode callee : callGraph.getPossibleTargets(node, call.getCallSite()))
          if (callee
              .getMethod()
              .getDeclaringClass()
              .getName()
              .toString()
              .equals("Ltensorflow/functions/reshape")) reshape = true;
        if (!reshape) continue;
        PointerKey key =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(node, call.getReturnValue(0));
        PointsToSetVariable variable = builder.getPropagationSystem().findOrCreatePointsToSet(key);
        TensorGenerator generator = TensorGeneratorFactory.getGenerator(variable, builder);
        if (generator == null) continue;
        ret.add(generator.getTensorTypes(builder));
      }
    }
    return ret;
  }

  @Test
  public void testReshapesReadDirectlyAreNeverBottom()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<Set<TensorType>> reads = readReshapesDirectly();
    // The two helpers' reshapes, once per context the layer call and the layer loop make.
    assertTrue("at least two reshape results were read: " + reads, reads.size() >= 2);
    for (Set<TensorType> read : reads)
      // A read of {@code null} is a tensor of unknown type (⊤); an empty set is "not a tensor"
      // (⊥), which only a deferral shipping on this path could produce for a reshape result.
      assertFalse(
          "a reshape result read as not a tensor: " + reads, read != null && read.isEmpty());
    // The direct call's reshape, `orig_dims + [width]`, reads its exact target on this path as it
    // did before, so (8, 10, 32) is among the results read.
    assertTrue(
        "the direct call's round trip reads (8, 10, 32): " + reads,
        reads.stream()
            .filter(r -> r != null)
            .anyMatch(r -> r.contains(TensorType.of(FLOAT_32, 8, 10, 32))));
  }
}
