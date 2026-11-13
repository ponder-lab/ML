package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code rejection_resample} calls its {@code class_func} on the receiver's element, the method and
 * {@code tf.data.experimental.rejection_resample} handed to {@code apply} alike, and yields {@code
 * (class, element)} pairs (<a href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetRejectionClassFunc extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_rejection_class_func.py";

  /**
   * {@code rejection_resample} calls its {@code class_func} on the receiver's element, so the
   * function's body is analyzed and its argument is the element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testClassFuncArgument() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_class_arg", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * {@code tf.data.experimental.rejection_resample} handed to {@code apply} calls the function the
   * same way.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedClassFuncArgument()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_applied_class_arg", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The method's pair's first member is the class, an {@code int32} scalar.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testClass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_class", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * The method's pair's second member is the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_element", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The applied function's pair's first member is the class, an {@code int32} scalar.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedClass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_applied_class", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * The applied function's pair's second member is the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_applied_element", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }
}
