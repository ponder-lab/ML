package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a slice of a Python list or tuple of tensors is not itself a tensor, while a slice of
 * a tensor is (wala/ML#1121). Before, the slice of a list was typed as a slice of the list stacked
 * into one tensor, though its points-to set is the new list the slice allocates.
 */
public class TestListSliceNotTensor extends AbstractTensorTest {

  private static final String FILE = "tf2_test_list_slice_not_tensor.py";

  /** A slice of a list of tensors is a list, so the function receiving it has no tensor. */
  @Test
  public void testListSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sub", 0, 0);
  }

  /** A slice of a tuple of tensors is a tuple, so the function receiving it has no tensor. */
  @Test
  public void testTupleSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple_sub", 0, 0);
  }

  /** The control: a slice of a tensor is a tensor, with the slice's shape. */
  @Test
  public void testTensorSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tensor_slice", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * The control for operations: TensorFlow converts a list argument by stacking it, so an operation
   * over a slice of a list of tensors is still a tensor of the operation's type.
   */
  @Test
  public void testConcatOverSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_concat", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /** {@code tf.stack} over a slice of a list of tensors is a tensor. */
  @Test
  public void testStackOverSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stack", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /** {@code tf.add_n} over a slice of a list of tensors is a tensor. */
  @Test
  public void testAddNOverSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_add_n", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** {@code tf.convert_to_tensor} of a slice of a list of tensors is a tensor. */
  @Test
  public void testConvertedSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_converted", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /** {@code tf.reduce_mean} over a slice of a list of tensors is a scalar tensor. */
  @Test
  public void testMeanOverSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }
}
