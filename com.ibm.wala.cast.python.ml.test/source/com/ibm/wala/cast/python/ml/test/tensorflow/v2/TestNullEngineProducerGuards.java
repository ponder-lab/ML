package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The producer self-recursion guards on the null-engine path (wala/ML#928). A generator read made
 * through the public surface, {@link TensorGeneratorFactory#getGenerator} and {@link
 * TensorGenerator#getTensorTypes}, with no analysis run in progress has no worklist resolver
 * installed, so the read recurses and the guards decide what a query that meets its own allocation
 * reads. Every test that goes through {@code performAnalysis} runs under the resolver, where the
 * query iterates instead and the dtype guards contribute nothing for the iteration to revise, so
 * this class builds the call graph and reads directly.
 *
 * <p>The fixture makes three variables depend on their own producer in a loop. On the SHAPE axis
 * the guard reads ⊤, and a manual generator reads its operands through the default set view, whose
 * contract is the resolvable subset (wala/ML#718), so the ⊤ is dropped one hop up; for the tiling
 * the concrete members exclude the program's value (tracked as wala/ML#931). On the DTYPE axis the
 * guard reads {@code UNKNOWN}, the marked remainder of a partial union (wala/ML#862): the read runs
 * once and ships as read (wala/ML#753), so the self-referential member cannot be assumed to share
 * its siblings' dtype. Each sink therefore reads its siblings' float32 beside an unknown twin. The
 * guard contributed nothing while an {@code UNKNOWN} member collapsed the union to {@code UNKNOWN}
 * and erased the float32; the union now keeps it, and these pins record the sound partial.
 */
public class TestNullEngineProducerGuards extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_loop_carried_producer_direct.py";

  private static final TensorType UNKNOWN_SHAPE_FLOAT32 = new TensorType(FLOAT_32, null);

  private static final TensorType UNKNOWN_SHAPE_UNKNOWN = new TensorType(UNKNOWN, null);

  /**
   * Builds the fixture's call graph without running the analysis and reads the argument of each
   * sink directly through its generator.
   */
  private Map<String, Set<TensorType>> readDirectly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles("");
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, pathFiles, FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph callGraph = builder.makeCallGraph(builder.getOptions());
    // Deliberately no performAnalysis: no resolver is installed, so the reads below take the
    // null-engine path through the public contract.
    Map<String, Set<TensorType>> ret = new HashMap<>();
    for (CGNode node : callGraph) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions()) {
        if (!(instruction instanceof PythonInvokeInstruction)) continue;
        PythonInvokeInstruction call = (PythonInvokeInstruction) instruction;
        if (call.getNumberOfPositionalParameters() < 2) continue;
        for (CGNode callee : callGraph.getPossibleTargets(node, call.getCallSite())) {
          String name = callee.getMethod().getDeclaringClass().getName().toString();
          for (String sink : List.of("consume_tile", "consume_concat", "consume_add")) {
            if (!name.endsWith("/" + sink)) continue;
            PointerKey key =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(node, call.getUse(1));
            PointsToSetVariable variable =
                builder.getPropagationSystem().findOrCreatePointsToSet(key);
            TensorGenerator generator = TensorGeneratorFactory.getGenerator(variable, builder);
            assertNotNull("the argument of " + sink + " has a generator", generator);
            ret.put(sink, generator.getTensorTypes(builder));
          }
        }
      }
    }
    return ret;
  }

  @Test
  public void testLoopCarriedTileReadDirectlyKeepsItsDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<TensorType> types = readDirectly().get("consume_tile");
    assertNotNull("the tiling's argument was read", types);
    // Three iterations take (2, 3) to (16, 3), and the final tiling is (32, 3). The read is
    // {(8, 3), (4, 3)}: the self-referential member is dropped one hop up by the default set view
    // (a partial result returns its resolvable subset), so the concrete members EXCLUDE the
    // program's value and no unknown mark reaches this read. That set is unsound at this read on
    // either guard value and is tracked as wala/ML#931; it is pinned here so a change is seen.
    // Each shape member keeps the siblings' float32 beside the unknown remainder the
    // self-referential member contributes on the dtype axis (wala/ML#862).
    assertEquals(
        "the shape members, the program's (32, 3) absent: " + types,
        Set.of(
            TensorType.of(FLOAT_32, 8, 3),
            TensorType.of(UNKNOWN, 8, 3),
            TensorType.of(FLOAT_32, 4, 3),
            TensorType.of(UNKNOWN, 4, 3)),
        types);
  }

  @Test
  public void testLoopCarriedConcatReadDirectlyIsAControl()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<TensorType> types = readDirectly().get("consume_concat");
    assertNotNull("the concatenation's argument was read", types);
    // The concatenation declines to an unknown shape when any operand member is missing, so the
    // shape reads unknown alone whatever the shape guard returns: a control that the direct read
    // reaches the producer, not a witness of the shape value. The dtype half is the tiling's:
    // float32 beside the self-referential member's unknown remainder (wala/ML#862).
    assertEquals(
        "unknown shape, float32 and unknown: " + types,
        Set.of(UNKNOWN_SHAPE_FLOAT32, UNKNOWN_SHAPE_UNKNOWN),
        types);
  }

  @Test
  public void testLoopCarriedAddReadDirectlyKeepsItsDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<TensorType> types = readDirectly().get("consume_add");
    assertNotNull("the addition's argument was read", types);
    // The shape is stable across iterations, so the program's (2, 3) is the one member; the
    // unknown mark from the shape guard does not reach this read (see the tiling), and the dtype
    // is float32 beside the self-referential member's unknown remainder (wala/ML#862).
    assertEquals(
        "(2, 3) float32 and unknown: " + types,
        Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(UNKNOWN, 2, 3)),
        types);
  }
}
