package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests what {@code tf.nest.map_structure} and {@code tf.nest.pack_sequence_as} return (<a
 * href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>): the nesting of the structure
 * they are given, with the function's results, or the flat sequence's members, in the leaves'
 * places. The results are not paired with the leaves they were computed from, so a place reads
 * every result. Before, each call returned nothing.
 */
public class TestNestResults extends AbstractTensorTest {

  private static final String FILE = "tf2_test_nest_results.py";

  private static final TensorType TENSOR_2_6_FLOAT32 = TensorType.of(FLOAT_32, 2, 6);

  private static final TensorType TENSOR_2_7_FLOAT32 = TensorType.of(FLOAT_32, 2, 7);

  private static final TensorType TENSOR_2_8_FLOAT32 = TensorType.of(FLOAT_32, 2, 8);

  private static final TensorType TENSOR_2_9_FLOAT32 = TensorType.of(FLOAT_32, 2, 9);

  private static final TensorType TENSOR_3_8_FLOAT32 = TensorType.of(FLOAT_32, 3, 8);

  private static final TensorType TENSOR_5_7_FLOAT32 = TensorType.of(FLOAT_32, 5, 7);

  private static final TensorType TENSOR_8_FLOAT32 = TensorType.of(FLOAT_32, 8);

  private static final TensorType TENSOR_9_FLOAT32 = TensorType.of(FLOAT_32, 9);

  private static final TensorType TENSOR_2_5_FLOAT32 = TensorType.of(FLOAT_32, 2, 5);

  /** A value of the dict the call returns for a dict. */
  @Test
  public void testDict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_dict", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32, TENSOR_3_FLOAT32)));
  }

  /** A member of the tuple the call returns for a tuple. */
  @Test
  public void testTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_tuple", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32, TENSOR_4_FLOAT32)));
  }

  /**
   * A member of the list the call returns for a list, by a subscript and by an iteration, and for a
   * list built with {@code append}, by a concatenation and by a repetition.
   */
  @Test
  public void testList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_list", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32, TENSOR_6_1_FLOAT32)));
    test(FILE, "of_iterated", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32, TENSOR_6_1_FLOAT32)));
    test(FILE, "of_appended", 1, 1, Map.of(2, Set.of(TENSOR_2_6_FLOAT32)));
    test(FILE, "of_concatenated", 1, 1, Map.of(2, Set.of(TENSOR_2_8_FLOAT32)));
    test(FILE, "of_repeated", 1, 1, Map.of(2, Set.of(TENSOR_3_8_FLOAT32)));
  }

  /**
   * A member of a tuple in the dict the call returns for a dict that holds a tuple, and the tuple
   * itself, which is a structure and no tensor: no result takes a structure's place.
   */
  @Test
  public void testNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "of_nested",
        1,
        1,
        Map.of(2, Set.of(TENSOR_7_FLOAT32, TENSOR_2_3_FLOAT32, TENSOR_3_4_FLOAT32)));
    test(FILE, "of_inner", 0, 0);
  }

  /**
   * A list in a list: the inner list the call returns, which is no tensor, and its member. One call
   * copies both lists, which share a class, and copies them apart.
   */
  @Test
  public void testSameClassNesting()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_same_inner", 0, 0);
    test(FILE, "of_same_deep", 1, 1, Map.of(2, Set.of(TENSOR_2_9_FLOAT32)));
  }

  /**
   * A field of the namedtuple instance the call returns for one, read by its name, and the same for
   * the instance a second call returns for the first call's.
   */
  @Test
  public void testNamedTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_namedtuple", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32, TENSOR_4_3_FLOAT32)));
    test(FILE, "of_twice", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32, TENSOR_4_5_FLOAT32)));
  }

  /** For a tensor the call returns the function's result itself. */
  @Test
  public void testTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_tensor", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * The call returns the function's results and not the values the function was given, the
   * structure it was given keeps its values, and the structure it returns is no tensor.
   */
  @Test
  public void testResultsReplaceLeaves()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_replaced", 1, 1, Map.of(2, Set.of(TENSOR_7_FLOAT32)));
    test(FILE, "of_source", 1, 1, Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
    test(FILE, "of_structure", 0, 0);
  }

  /**
   * The result of a function of two structures has the first structure's nesting. The function is
   * also called with a leaf of the first structure alone, so the sum of its two parameters reads a
   * tensor of unknown shape beside the leaves' type. The control's function returns its first
   * parameter, and reads the leaf's type alone.
   */
  // TODO: Expect (5, 6) alone once a function of two structures is called with a leaf of each
  // only (wala/ML#1287).
  @Test
  public void testTwoStructures()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_two", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32)));
    test(FILE, "of_first", 1, 1, Map.of(2, Set.of(TENSOR_5_7_FLOAT32)));
  }

  /** The same for a structure that is a function's parameter. */
  @Test
  public void testParameter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_parameter", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * A {@code None} is a leaf: the function's result takes its place, which is a tensor where the
   * function returns one and no tensor where it returns the {@code None}.
   */
  @Test
  public void testNoneLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_none", 1, 1, Map.of(2, Set.of(TENSOR_8_FLOAT32)));
    test(FILE, "of_none_place", 0, 0);
  }

  /**
   * A structure the analysis reads nothing for has no rebuilt nesting, and no result takes a place
   * in it.
   */
  // TODO: Expect (6,) once a structure with no points-to set gets its function's results
  // (wala/ML#1280).
  @Test
  public void testUnreadStructure()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_unread", 0, 0);
  }

  /**
   * What {@code tf.nest.pack_sequence_as} returns: a value of the dict it returns for a dict, the
   * dict itself, which is no tensor, the sequence's member for a structure that is a leaf, and a
   * value of the dict it returns for the list {@code tf.nest.flatten} returns.
   */
  @Test
  public void testPackSequenceAs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_packed", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_3_1_FLOAT32)));
    test(FILE, "of_packed_structure", 0, 0);
    test(FILE, "of_packed_tensor", 1, 1, Map.of(2, Set.of(TENSOR_9_FLOAT32)));
    test(FILE, "of_repacked", 1, 1, Map.of(2, Set.of(TENSOR_2_7_FLOAT32)));
  }

  /** The element of a dataset whose {@code map} function returns the call's result. */
  @Test
  public void testDatasetElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_element", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** The result for a tensor an operator computed. */
  @Test
  public void testOperatorLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "of_product", 1, 1, Map.of(2, Set.of(TENSOR_2_5_FLOAT32)));
  }
}
