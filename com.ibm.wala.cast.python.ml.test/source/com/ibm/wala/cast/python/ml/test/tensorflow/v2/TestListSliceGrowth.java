package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A slice of a list literal holds the literal's elements in range only while nothing may have
 * changed the list: once the program may grow it by a route the analysis does not record among its
 * positioned elements, the slice holds every element the literal had (<a
 * href="https://github.com/wala/ML/issues/1250">wala/ML#1250</a>). Each list is {@code [a, h, a]}
 * when it is sliced, so {@code [0:-1]} is {@code [a, h]}, whose concatenation has six rows; a slice
 * trimmed to the literal's {@code [a, h][0:-1]} holds {@code a} alone, of four. A list grown by
 * {@code append} keeps its literal's positions, and its slice its six rows. Any other route may
 * also move the literal's elements, so the slice holds them at positions it does not know, and its
 * concatenation has an unknown number of rows, but never four.
 */
public class TestListSliceGrowth extends AbstractTensorTest {

  private static final String FILE = "tf2_test_list_slice_growth.py";

  /** The concatenation of {@code [a, h]}. */
  private static final TensorType TENSOR_6_3_FLOAT32 = TensorType.of(FLOAT_32, 6, 3);

  /**
   * A list grown by {@code append}, which the analysis records in the list's appended contents, the
   * control.
   */
  @Test
  public void testAppended() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_appended", 1, 1, Map.of(2, Set.of(TENSOR_6_3_FLOAT32)));
  }

  /**
   * A list grown by {@code extend}.
   *
   * <p>TODO: <a href="https://github.com/wala/ML/issues/1070">wala/ML#1070</a>: once {@code extend}
   * is modeled as {@code append} is, {@link #TENSOR_6_3_FLOAT32}.
   */
  @Test
  public void testExtended() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_extended", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** A list grown by {@code insert}. */
  @Test
  public void testInserted() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inserted", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A list grown by {@code +=}, whose result holds none of the elements it adds, so its slice has
   * neither a shape nor a dtype.
   *
   * <p>TODO: <a href="https://github.com/wala/ML/issues/1069">wala/ML#1069</a>: once the result
   * holds them, {@link #TENSOR_UNKNOWN_SHAPE_FLOAT32}.
   */
  @Test
  public void testAugmented() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_augmented", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /** A list grown by a slice assignment. */
  @Test
  public void testAssigned() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_assigned", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** A list grown by {@code operator.iadd}, a call the analysis resolves to no target. */
  @Test
  public void testOperator() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_operator", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
