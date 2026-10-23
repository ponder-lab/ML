package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code fit} on a {@code tf.data.Dataset} with Keras's default {@code train_step} feeds
 * the model's {@code call} the element's first component (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>), not the whole element with its
 * targets, and that {@code call} reads within that component when it is a tuple.
 */
public class TestFitDefaultStep extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_default_step.py";

  /** A rank-1 {@code int32} tensor of unknown extent. */
  private static final TensorType TENSOR_NONE_INT32 =
      new TensorType(INT_32, asList(DynamicDim.INSTANCE));

  /** The model's inputs are exactly the element's first component, a tensor. */
  @Test
  public void testFlatInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_inputs", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /** A component of the inputs, when the inputs are a tuple, is that component alone. */
  @Test
  public void testNestedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_nested_index", 1, 1, Map.of(2, Set.of(TENSOR_NONE_INT32)));
  }

  /** As {@link #testNestedInputs()}, through a distribution strategy's dataset. */
  @Test
  public void testDistributedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_distributed_index", 1, 1, Map.of(2, Set.of(TENSOR_NONE_INT32)));
  }

  /**
   * As {@link #testNestedInputs()}, through a {@code train_step} override's {@code x, y = data}.
   */
  @Test
  public void testOverrideInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_override_index", 1, 1, Map.of(2, Set.of(TENSOR_NONE_INT32)));
  }

  /**
   * A model class fit twice, on datasets whose inputs differ in dtype, has both inputs in {@code
   * call}, each batch's full and remainder extents.
   */
  @Test
  public void testTwiceFitInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_twice_inputs",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                TensorType.of(FLOAT_32, 1, 4),
                TensorType.of(INT_32, 2, 4),
                TensorType.of(INT_32, 1, 4))));
  }

  /**
   * A dict component of the inputs, read by key in {@code call}, is that key's value (<a
   * href="https://github.com/wala/ML/issues/1002">wala/ML#1002</a>).
   */
  @Test
  public void testDictInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dict_ids", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 4, 3))));
  }

  /**
   * A tuple of inputs, {@code (x, mask)} in a {@code ((x, mask), y)} element of a batched {@code
   * from_tensor_slices} dataset, reaches {@code call} as one batch of each member: {@code (2, 4)}
   * and {@code (2, 3)}, each member sliced along its own leading axis, and not the members stacked
   * into one tensor whose stacking axis is sliced, which read {@code (2, 8, 4)} and {@code (2, 8,
   * 3)}.
   */
  @Test
  public void testTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_structured_inputs.py",
        "consume_pair",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A member of the tuple inputs of {@link #testTupleInputs()}, read in {@code call}. */
  @Test
  public void testTupleInputsMember() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_structured_inputs.py",
        "consume_pair_x",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A dict of inputs, {@code {"ids": ids}} in a {@code ({"ids": ids}, y)} element, reaches {@code
   * call} as one batch of its value, {@code (4, 3) int32}, and not a join with the targets' types
   * and an unknown.
   */
  @Test
  public void testDictInputsWhole() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_structured_inputs.py",
        "consume_dict",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 4, 3))));
  }
}
