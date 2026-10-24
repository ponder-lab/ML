package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@code tf.ensure_shape}, whose result is its input with the declared static shape, and
 * {@code tf.keras.backend.int_shape}, whose result is a Python tuple of the input's static extents
 * and never a tensor.
 */
public class TestEnsureShapeIntShape extends AbstractTensorTest {

  private static final String FILE = "tf2_test_ensure_shape_int_shape.py";

  /**
   * An input whose shape is already the declared one keeps it, and its dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEnsureShapeKnown() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_known", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A {@code None} in the declared shape keeps the input's extent at that position.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEnsureShapeNoneKeepsInputExtent()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_none_dim", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The control for {@link #testEnsureShapeRefinesDynamicExtent()}: the Keras input itself has a
   * dynamic batch extent.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDynamicBatchControl()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_batch", 1, 1, Map.of(2, Set.of(TENSOR_NONE_3_FLOAT32)));
  }

  /**
   * A declared extent refines an input's dynamic one, as a Keras input of shape {@code (None, 3)}
   * ensured to {@code (2, 3)} is.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEnsureShapeRefinesDynamicExtent()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_refined", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A {@code None} over a dynamic input extent stays dynamic, as for the same input ensured to
   * {@code (None, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEnsureShapeNoneOverDynamicExtent()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dynamic", 1, 1, Map.of(2, Set.of(TENSOR_NONE_3_FLOAT32)));
  }

  /**
   * A list of Python numbers added on the left of an ensured float32 tensor is converted to the
   * tensor's dtype, as {@code t.__radd__} converts it, rather than imposing its integers' int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumberListPlusTensorTakesTensorDType()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_list_plus", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 2, 3))));
  }

  /**
   * A list of Python numbers added on the left of a partner that is a TensorFlow tensor on one path
   * and a NumPy array on another takes the tensor's dtype on the first and NumPy's promotion on the
   * second: the list becomes an int64 array, which promotes with float32 to float64.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumberListPlusMixedPartner()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_mixed_partner",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TensorType.of(FLOAT_64, 2, 3))));
  }

  /**
   * A list of Python numbers added on the left of a Keras layer call's result takes the result's
   * float32, not its integers' int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumberListPlusLayerOutput()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_layer_plus", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * As {@link #testNumberListPlusLayerOutput()}, beside an element-wise result, {@code known *
   * 2.0}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumberListPlusElementwiseResult()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_binop_plus", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * As {@link #testNumberListPlusLayerOutput()}, beside the result of a call of a user layer.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumberListPlusUserLayerOutput()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_user_layer_plus", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A list of Python bools added on the left of a float32 tensor is converted to float32, as a list
   * of numbers is.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBoolListPlusTensorTakesTensorDType()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bool_list_plus", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * An input that only a sidecar annotation types, whose points-to set is empty, reaches the result
   * through the dataflow, refined by the declared shape: {@code (2, 3)} ensured to {@code (2,
   * None)} stays {@code (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEnsureShapeOfSidecarTypedInput()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {"sidecar_ensure_proj/loader.py", "sidecar_ensure_proj/driver.py"},
        "driver.py",
        "consume_ensured",
        "sidecar_ensure_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The tuple {@code int_shape} returns is no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIntShapeIsNoTensor()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_int_shape", 0, 0);
  }

  /**
   * An element of the tuple {@code int_shape} returns, a Python int, is no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIntShapeElementIsNoTensor()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_int_shape_element", 0, 0);
  }

  /**
   * An allocation shaped by {@code int_shape(t)} takes {@code t}'s shape, as one shaped by {@code
   * t.shape} does.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAllocationShapedByIntShape()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zeros", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
