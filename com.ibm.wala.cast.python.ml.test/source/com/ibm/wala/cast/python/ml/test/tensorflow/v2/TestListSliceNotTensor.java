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
}
