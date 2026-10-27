package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Reads made outside an analysis run of values whose producers depend on each other in a cycle (<a
 * href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>). Such a read, made through the
 * public surface ({@link TensorGeneratorFactory#getGenerator} and {@link
 * TensorGenerator#getTensorTypes}) after the analysis, as a client reading each function's types
 * does, has no worklist resolver installed. The resolver converges a cycle of queries; without it,
 * a cycle through two operations, two nodes, or two φs recursed until the stack overflowed, since
 * the only guards stopped a producer reading its own allocation in its own node. A query met again
 * while it is being evaluated now reads as unknown, the resolver's own answer for a cycle.
 *
 * <p>Each fixture function mirrors a program shape that overflowed: a NumPy array and a cast fed
 * back through a constructor parameter, and a value assigned under a branch in a loop, so that the
 * loop's φ and the branch's φ read each other. The two controls, a cast and a concatenation fed
 * back through a constructor, and a dataset zipped with itself in a loop, do not overflow without
 * the guard.
 */
public class TestNullEngineProducerCycles extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_null_engine_producer_cycles.py";

  private static final TensorType TENSOR_2_3_INT32 = TensorType.of(INT_32, 2, 3);

  private static final TensorType TENSOR_NONE_3_INT32 =
      new TensorType(INT_32, List.of(DynamicDim.INSTANCE, new NumericDim(3)));

  private static final TensorType TENSOR_3_FLOAT32 = TensorType.of(FLOAT_32, 3);

  private static final TensorType UNKNOWN_SHAPE_UNKNOWN = new TensorType(UNKNOWN, null);

  /**
   * Builds the fixture's call graph without running the analysis and reads the argument of the
   * given sink directly through its generator.
   *
   * @param sink The sink function's name.
   * @return The argument's types.
   */
  private Set<TensorType> readDirectly(String sink)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles("");
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, pathFiles, FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph callGraph = builder.makeCallGraph(builder.getOptions());
    // Deliberately no performAnalysis: no resolver is installed, so the read below takes the
    // null-engine path through the public contract.
    for (CGNode node : callGraph) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions()) {
        if (!(instruction instanceof PythonInvokeInstruction)) continue;
        PythonInvokeInstruction call = (PythonInvokeInstruction) instruction;
        if (call.getNumberOfPositionalParameters() < 2) continue;
        for (CGNode callee : callGraph.getPossibleTargets(node, call.getCallSite())) {
          String name = callee.getMethod().getDeclaringClass().getName().toString();
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
          return generator.getTensorTypes(builder);
        }
      }
    }
    return null;
  }

  /**
   * Runs the analysis over the fixture and collects each sink's parameter types.
   *
   * @return Each sink's parameter types, by the sink's name.
   */
  private Map<String, Set<TensorType>> readInsideAnalysis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles("");
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, pathFiles, FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    Map<String, Set<TensorType>> ret = new HashMap<>();
    analysis.forEach(
        pt -> {
          // A sink's one parameter is its second value; the first is the function object.
          if (!(pt.fst instanceof LocalPointerKey lpk) || lpk.getValueNumber() != 2) return;
          String method = lpk.getNode().getMethod().getDeclaringClass().getName().toString();
          String sink = method.substring(method.lastIndexOf('/') + 1);
          if (!sink.startsWith("consume_") || pt.snd == null || pt.snd.getTypes() == null) return;
          ret.computeIfAbsent(sink, k -> new HashSet<>()).addAll(pt.snd.getTypes());
        });
    return ret;
  }

  /**
   * A NumPy array and a cast that feed each other through a constructor parameter, read outside an
   * analysis run (<a href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>). The program's
   * value is {@code (2, 3)} float64. The read keeps the shape with an unknown dtype, where the
   * analysis reads float32 beside the unknown.
   */
  @Test
  public void testArrayAndCastCycleReadDirectly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(Set.of(TensorType.of(UNKNOWN, 2, 3)), readDirectly("consume_array"));
  }

  /**
   * A cast reading, by value number, a loop's φ that the φ of a branch in the loop reads back, read
   * outside an analysis run (<a href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>):
   * {@code (2, 3)} int32, the program's value and the analysis's.
   */
  @Test
  public void testRecastPhiCycleReadDirectly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(Set.of(TENSOR_2_3_INT32), readDirectly("consume_recast"));
  }

  /**
   * A concatenation and a cast under a branch in a loop, whose φs read each other, read outside an
   * analysis run (<a href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>): a leading
   * axis that grows with the loop, so {@code (None, 3)} int32, as the analysis reads it. The
   * program's last value is {@code (6, 3)}.
   */
  @Test
  public void testLoopPhiCycleReadDirectly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(Set.of(TENSOR_NONE_3_INT32), readDirectly("consume_grown"));
  }

  /**
   * A control: a cast and a concatenation that feed each other through a constructor parameter, a
   * program shape that did not overflow before the cycle guard (<a
   * href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>). The read gives the first
   * construction's {@code (2, 3)} int32 alone, where the analysis also has an unknown shape.
   */
  @Test
  public void testCastAndConcatenationReadDirectlyIsAControl()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(Set.of(TENSOR_2_3_INT32), readDirectly("consume_cast"));
  }

  /**
   * A control: an element of a dataset zipped with itself and mapped in a loop, read outside an
   * analysis run, a program shape that did not overflow before the cycle guard (<a
   * href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>): the dataset generators' own
   * reads of their receivers terminate. The program's element is {@code (3,)} float32; the read is
   * that beside an unknown, as the analysis reads it.
   */
  @Test
  public void testDatasetZipReadDirectlyIsAControl()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(Set.of(TENSOR_3_FLOAT32, UNKNOWN_SHAPE_UNKNOWN), readDirectly("consume_zipped"));
  }

  /**
   * The same values read inside an analysis run, where the resolver converges the cycles (<a
   * href="https://github.com/wala/ML/issues/1174">wala/ML#1174</a>).
   */
  @Test
  public void testCyclesInsideAnalysis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Map<String, Set<TensorType>> types = readInsideAnalysis();
    assertEquals(
        Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(UNKNOWN, 2, 3)),
        types.get("consume_array"));
    assertEquals(Set.of(TENSOR_2_3_INT32), types.get("consume_recast"));
    assertEquals(Set.of(TENSOR_NONE_3_INT32), types.get("consume_grown"));
    assertEquals(Set.of(TENSOR_2_3_INT32, new TensorType(INT_32, null)), types.get("consume_cast"));
    assertEquals(Set.of(TENSOR_3_FLOAT32, UNKNOWN_SHAPE_UNKNOWN), types.get("consume_zipped"));
  }
}
