package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.RebuiltStructureKey;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A {@code tf.nest.map_structure} or {@code tf.nest.pack_sequence_as} call whose result reaches its
 * own structure argument, as in a loop that reassigns a variable to the call's result, still has a
 * finite number of copies of the structure, so the call graph is built (<a
 * href="https://github.com/wala/ML/issues/1297">wala/ML#1297</a>). Before, the call copied its own
 * copies without end.
 */
public class TestNestRebuildCycles extends AbstractTensorTest {

  private static final String FILE = "tf2_test_nest_rebuild_cycles.py";

  private static final TensorType TENSOR_2_8_FLOAT32 = TensorType.of(FLOAT_32, 2, 8);

  private static final TensorType TENSOR_9_FLOAT32 = TensorType.of(FLOAT_32, 9);

  private static final TensorType TENSOR_5_7_FLOAT32 = TensorType.of(FLOAT_32, 5, 7);

  private static final TensorType TENSOR_3_8_FLOAT32 = TensorType.of(FLOAT_32, 3, 8);

  private static final TensorType TENSOR_2_9_FLOAT32 = TensorType.of(FLOAT_32, 2, 9);

  private static final TensorType TENSOR_6_2_FLOAT32 = TensorType.of(FLOAT_32, 6, 2);

  private static final TensorType TENSOR_6_3_FLOAT32 = TensorType.of(FLOAT_32, 6, 3);

  private static PythonSSAPropagationCallGraphBuilder builder;

  /**
   * Builds the fixture's call graph, once.
   *
   * @return Its builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestNestRebuildCycles().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * The copies among the structures a sink receives.
   *
   * @param sink The sink's name. It has one node.
   * @return The rebuilt structures in the points-to set of its parameter.
   */
  private static List<RebuiltStructureKey> copies(String sink) {
    List<PointerKey> parameters = new ArrayList<>();
    for (CGNode node : builder().getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
        parameters.add(
            builder().getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for " + sink + ".", 1, parameters.size());
    List<RebuiltStructureKey> ret = new ArrayList<>();
    for (InstanceKey key : builder().getPointerAnalysis().getPointsToSet(parameters.get(0)))
      if (key instanceof RebuiltStructureKey copy) ret.add(copy);
    return ret;
  }

  /**
   * The call graph and the tensor analysis finish on the fixture.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test(timeout = 120_000)
  public void testFinishes() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
  }

  /**
   * A variable a loop reassigns to a call's result holds one copy of its structure after the loop,
   * made by the call whose result the loop leaves in it, the second of two calls that feed each
   * other. No copy is a copy of a copy.
   */
  @Test(timeout = 120_000)
  public void testCopies() {
    assertEquals(1, copies("holds_looped").size());
    assertEquals(1, copies("holds_repacked").size());
    assertEquals(1, copies("holds_alternated").size());
    assertEquals(1, copies("holds_named").size());
    assertEquals(1, copies("holds_deepened").size());
    for (String sink :
        List.of(
            "holds_looped", "holds_repacked", "holds_alternated", "holds_named", "holds_deepened"))
      for (RebuiltStructureKey copy : copies(sink))
        assertEquals(
            "A copy of a copy in " + sink + ".",
            false,
            copy.getSource() instanceof RebuiltStructureKey);
  }

  /**
   * A member of the tuple one call in a loop returns, and a member of the list the tuple holds. The
   * results are not paired with the leaves they were computed from, so a place reads every result.
   */
  @Test(timeout = 120_000)
  public void testLooped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<TensorType> results = Set.of(TENSOR_2_3_FLOAT32, TENSOR_4_FLOAT32, TENSOR_5_FLOAT32);
    test(FILE, "of_looped", 1, 1, Map.of(2, results));
    test(FILE, "of_looped_inner", 1, 1, Map.of(2, results));
  }

  /** A member of a tuple packed around its own leaves in a loop. */
  @Test(timeout = 120_000)
  public void testRepacked()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_repacked", 1, 1, Map.of(2, Set.of(TENSOR_3_6_FLOAT32, TENSOR_7_FLOAT32)));
  }

  /** A value of the dict two calls in a loop return, each given the other's result. */
  @Test(timeout = 120_000)
  public void testAlternated()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_alternated", 1, 1, Map.of(2, Set.of(TENSOR_2_8_FLOAT32, TENSOR_9_FLOAT32)));
  }

  /**
   * A field of the namedtuple instance a call in a loop returns, read by its name: a copy has the
   * names of the instance it ultimately copies.
   */
  @Test(timeout = 120_000)
  public void testNamed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_named", 1, 1, Map.of(2, Set.of(TENSOR_5_7_FLOAT32, TENSOR_3_8_FLOAT32)));
  }

  /**
   * A leaf of the result of a function that returns a structure, mapped in a loop, so every turn
   * nests the result once more.
   */
  @Test(timeout = 120_000)
  public void testDeepened()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_deepened", 1, 1, Map.of(2, Set.of(TENSOR_2_9_FLOAT32)));
  }

  /**
   * A control with no loop: the copy the second of two calls makes of the first's copy holds the
   * second function's results alone.
   */
  @Test(timeout = 120_000)
  public void testChained()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_once", 1, 1, Map.of(2, Set.of(TENSOR_6_2_FLOAT32, TENSOR_6_3_FLOAT32)));
    test(FILE, "of_twice", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }
}
