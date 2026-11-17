package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the function {@code tf.nest.map_structure} is given is called with the leaves of its
 * structure, so its parameter is typed from them, and that {@code tf.nest.flatten} returns a list
 * of the leaves (<a href="https://github.com/wala/ML/issues/1067">wala/ML#1067</a>). Before, no
 * summary modeled {@code tf.nest}, and a function called only through {@code map_structure} had no
 * call-graph node.
 */
public class TestNestStructures extends AbstractTensorTest {

  private static final String FILE = "tf2_test_nest_structures.py";

  private static final TensorType TENSOR_UNRESOLVED_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(3)));

  private static final TensorType TENSOR_1_2_3_FLOAT32 = TensorType.of(FLOAT_32, 1, 2, 3);

  /** The values of a dict are its leaves, whatever its keys. */
  @Test
  public void testDict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_dict", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32, TENSOR_3_INT32)));
  }

  /** The members of a tuple are its leaves. */
  @Test
  public void testTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_tuple", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32, TENSOR_4_FLOAT32)));
  }

  /** The members of a list literal are its leaves. */
  @Test
  public void testList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_list", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32, TENSOR_6_1_FLOAT32)));
  }

  /**
   * A dict holding a tuple, which holds a list, and another dict: the leaves are read through each
   * level.
   */
  @Test
  public void testNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "on_nested",
        1,
        1,
        Map.of(2, Set.of(TENSOR_7_FLOAT32, TENSOR_2_3_FLOAT32, TENSOR_3_4_FLOAT32)));
  }

  /** A tensor is a structure of one leaf, itself. */
  @Test
  public void testTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_tensor", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The fields of a namedtuple instance are its leaves. */
  @Test
  public void testNamedTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_namedtuple", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32, TENSOR_4_3_FLOAT32)));
  }

  /** The module imported as {@code tensorflow.python.util.nest} is the same one. */
  @Test
  public void testPrivateModulePath()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_private_path", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  /** A lambda is called as a named function is, here seen by the function it calls. */
  @Test
  public void testLambda()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "from_lambda", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /** The members a list was given by {@code append} are its leaves. */
  @Test
  public void testAppendedList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_appended", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32)));
  }

  /** Given two structures, the function takes a leaf of each. */
  @Test
  public void testTwoStructures()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "on_two",
        2,
        2,
        Map.of(2, Set.of(TENSOR_5_6_FLOAT32), 3, Set.of(TensorType.of(FLOAT_32, 3, 6))));
  }

  /** Given three structures, the function takes a leaf of each. */
  @Test
  public void testThreeStructures()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "on_three",
        3,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 7, 1)),
            3,
            Set.of(TensorType.of(FLOAT_32, 7, 2)),
            4,
            Set.of(TensorType.of(FLOAT_32, 7, 3))));
  }

  /** A keyword argument of the call is not a structure. */
  @Test
  public void testKeywordArgument()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_keyword", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 2))));
  }

  /**
   * The function of the fixture whose call-graph node {@code TestNestMapStructure} asserts: its
   * parameter holds both values of the dict it is mapped over, and its other local is the array
   * {@code numpy} returns.
   */
  @Test
  public void testMappedFunctionParameter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nest_map_structure.py",
        "to_numpy",
        1,
        2,
        Map.of(2, Set.of(TENSOR_2_INT32, TENSOR_2_FLOAT32)));
  }

  /**
   * A parameter the call does not pass keeps its default: the calls that pass a leaf of a second
   * and of a third structure have no target here, so the tensor sized by the default has that size
   * alone.
   */
  @Test
  public void testDefaultParameter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_default", 1, 2, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
    test(FILE, "sized", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** A variadic function's pack holds the leaf. */
  @Test
  public void testVariadicFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "star_first", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  /**
   * An integer subscript of the leaf, passed on. It is not typed today, as it is where the function
   * is called directly. The function is given a leaf and never the list holding it: a subscript of
   * the list would read the leaf whole, as {@code [2, 3]}.
   */
  // TODO: https://github.com/wala/ML/issues/1284; `member_row` then has one tensor parameter and
  // one tensor local, and the parameter reads `TENSOR_3_FLOAT32`.
  @Test
  public void testSubscriptOfLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "member_row", 0, 0, Map.of());
  }

  /** A slice of the leaf, passed on. Its leading size is not computed today. */
  // TODO: https://github.com/wala/ML/issues/1284; the parameter then reads `TENSOR_1_3_FLOAT32`.
  @Test
  public void testSliceOfLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "member_rows", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_3_FLOAT32)));
  }

  /** The function reading both has two tensor locals: its parameter and the slice. */
  // TODO: https://github.com/wala/ML/issues/1284; the subscript is then a third.
  @Test
  public void testSubscriptedLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "on_member", 1, 2, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A reshape of the leaf reads the leaf's type. */
  @Test
  public void testReshapeOfLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "reshaped", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /** A binary operator on the leaf reads the leaf's type. */
  @Test
  public void testBinaryOperatorOnLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "summed", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** An {@code expand_dims} of the leaf reads the leaf's type. */
  @Test
  public void testExpandDimsOfLeaf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "expanded", 1, 1, Map.of(2, Set.of(TENSOR_1_2_3_FLOAT32)));
  }

  /** A subscript of the list {@code flatten} returns reads a leaf. */
  @Test
  public void testFlattenSubscript()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "flat_subscript", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32, TENSOR_4_5_FLOAT32)));
  }

  /** An iteration over the list {@code flatten} returns reads the leaves. */
  @Test
  public void testFlattenIteration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "flat_iteration", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32, TENSOR_4_5_FLOAT32)));
  }
}
