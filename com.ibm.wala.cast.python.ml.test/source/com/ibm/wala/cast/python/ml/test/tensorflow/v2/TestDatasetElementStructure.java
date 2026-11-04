package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The element of a {@code tf.data.Dataset} built from a structure is an object whose positions are
 * fields, each holding its member (<a
 * href="https://github.com/wala/ML/issues/1190">wala/ML#1190</a>). A tuple or a dict element, or a
 * component that is one, is no tensor; its members keep their own shapes and dtypes; and a member
 * read off the element points to the member's own object.
 */
public class TestDatasetElementStructure extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_element_structure.py";

  /** A tuple element bound by a loop is no tensor. */
  @Test
  public void testLoopElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_element", 0, 0, Map.of());
  }

  /** A tuple component bound by unpacking a loop's element is no tensor. */
  @Test
  public void testLoopInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_inputs", 0, 0, Map.of());
  }

  /** The first member of the tuple component. */
  @Test
  public void testLoopX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** The second member of the tuple component. */
  @Test
  public void testLoopMask() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_mask", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The element's second position, a tensor. */
  @Test
  public void testLoopY() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_y", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /** A tuple element read by {@code next} is no tensor. */
  @Test
  public void testNextElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_next_element", 0, 0, Map.of());
  }

  /**
   * A dict component is no tensor, and in particular not the cross product of its members' shapes
   * and dtypes (<a href="https://github.com/wala/ML/issues/1138">wala/ML#1138</a>).
   */
  @Test
  public void testKeyedFeatures() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_features", 0, 0, Map.of());
  }

  /** The {@code "a"} entry of the dict component. */
  @Test
  public void testKeyedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** The {@code "b"} entry of the dict component, with its own dtype. */
  @Test
  public void testKeyedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /** The {@code "a"} entry of a dict element read by {@code next}. */
  @Test
  public void testFlatKeyedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_keyed_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** A list source is stacked into one tensor, not a structure. */
  @Test
  public void testStackedList() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stacked", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** Transforms that leave the element unchanged keep its structure. */
  @Test
  public void testShuffledMask() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_shuffled_mask", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * A tuple component of an element mapped by user code and then batched is no tensor: the mapped
   * element is the function's return, which the batch copies
   * (https://github.com/wala/ML/issues/164).
   */
  @Test
  public void testMappedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mapped_inputs", 0, 0, Map.of());
  }

  /** A member of the mapped and batched element's tuple component. */
  @Test
  public void testMappedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mapped_b", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * A value that is a dataset's element on one arm and an ordinary tensor on the other keeps both
   * arms' types through an operation over it.
   */
  @Test
  public void testEitherCast() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_either_cast",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 4), TensorType.of(INT_32, 3, 5))));
  }

  /**
   * A member nested past the depth the element's copy records reads as unknown, not as no tensor.
   */
  @Test
  public void testDeepMember() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_deep", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /** A tuple member read off a loop's element, cast and then scaled. */
  @Test
  public void testMemberCastScaled() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_member_cast_scaled", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 4))));
  }

  /** A dict member read off a loop's element, cast and then divided. */
  @Test
  public void testKeyedMemberCastScaled()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_keyed_member_cast_scaled",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A dict member read off an element {@code next} reads, plus a constant. */
  @Test
  public void testNextMemberShifted() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_next_member_shifted",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * Whether the parameter of the named sink points to anything, in every node of the sink.
   *
   * @param builder The builder.
   * @param sink The sink's name.
   * @return Whether each node's parameter has a non-empty points-to set.
   */
  private static boolean pointsToSomething(
      PythonSSAPropagationCallGraphBuilder builder, String sink) {
    boolean found = false;
    for (CGNode node : builder.getCallGraph()) {
      if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) continue;
      found = true;
      PointerKey parameter =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      if (builder.getPointerAnalysis().getPointsToSet(parameter).isEmpty()) return false;
    }
    return found;
  }

  /**
   * A member read off an element points to the member's own object, so a points-to client finds it
   * by the element's fields rather than only by its tensor type.
   */
  @Test
  public void testMembersPointToObjects() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    for (String sink :
        new String[] {
          "consume_keyed_a", "consume_keyed_b", "consume_flat_keyed_a", "consume_loop_x"
        })
      assertTrue(sink + "'s parameter points to the member.", pointsToSomething(builder, sink));
    assertFalse(
        "A sink the program never calls has no node.", pointsToSomething(builder, "absent"));
  }

  /**
   * A dict member whose key is digits too long for an {@code int}: the key is no tuple position.
   */
  @Test
  public void testLongDigitKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_long_digit_key", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /**
   * A dict member whose key is digits, {@code "0"}: the key is no tuple position (<a
   * href="https://github.com/wala/ML/issues/1198">wala/ML#1198</a>).
   */
  @Test
  public void testDigitKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_digit_key", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** A tuple element's member read at position 0, the digit key's positional twin. */
  @Test
  public void testTuplePosition() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple_position", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }
}
