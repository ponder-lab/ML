package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the element of a dataset of dicts, or of tuples of dicts, is typed when read in a loop
 * over the dataset (<a href="https://github.com/wala/ML/issues/993">wala/ML#993</a>): the
 * subscripts select a path into the dataset's element structure, by integer index into a tuple and
 * by string key into a dict, and the component's shape is the sliced tensor's with its first axis
 * dropped, batched where the dataset is batched. Before, a string key resolved nothing and the
 * element read as a tensor of unknown shape and dtype.
 */
public class TestDatasetDictElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_dict_element.py";

  private static final TensorType IDS_3 = TensorType.of(INT_32, 3);

  private static final TensorType IDS_2_3 = TensorType.of(INT_32, 2, 3);

  @Test
  public void testDict() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dict", 1, 1, Map.of(2, Set.of(IDS_3)));
  }

  @Test
  public void testDictBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dict_batched", 1, 1, Map.of(2, Set.of(IDS_2_3)));
  }

  /** Control: a tuple of tensors, selected by integer index, was typed before. */
  @Test
  public void testTupleOfTensors() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple_tensor", 1, 1, Map.of(2, Set.of(IDS_3)));
  }

  @Test
  public void testTupleOfDicts() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple_dict", 1, 1, Map.of(2, Set.of(IDS_3)));
  }

  @Test
  public void testTupleOfDictsBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple_dict_batched", 1, 1, Map.of(2, Set.of(IDS_2_3)));
  }

  /**
   * The same element passed into a function and subscripted there, whose signature is the dataset's
   * {@code element_spec}: the dict parameter's subscript is not seeded inside the callee, through
   * the signature ({@code element_spec} is unmodeled) or through the loop's call. This is
   * wala/ML#993's remainder on this chain.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#993 is fully fixed.
   */
  @Test(expected = AssertionError.class)
  public void testElementSpecStep() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_spec_step", 1, 1, Map.of(2, Set.of(IDS_2_3)));
  }
}
