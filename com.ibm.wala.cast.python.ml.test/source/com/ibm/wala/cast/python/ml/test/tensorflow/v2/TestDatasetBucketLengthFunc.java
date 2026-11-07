package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code bucket_by_sequence_length} calls its {@code element_length_func} on the receiver's
 * element, the method and {@code tf.data.experimental.bucket_by_sequence_length} handed to {@code
 * apply} alike (<a href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetBucketLengthFunc extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_bucket_length_func.py";

  /**
   * {@code bucket_by_sequence_length} calls its {@code element_length_func} on the receiver's
   * element, so the function's body is analyzed and its argument is the element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testLengthFuncArgument()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_length_arg", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * {@code tf.data.experimental.bucket_by_sequence_length} handed to {@code apply} calls the
   * function the same way.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedLengthFuncArgument()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_applied_length_arg", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A pair element is spread into the function's arguments: the first is the pair's first member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPairLengthFuncX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_length_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The second argument is the pair's second member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPairLengthFuncY() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_length_y", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }
}
