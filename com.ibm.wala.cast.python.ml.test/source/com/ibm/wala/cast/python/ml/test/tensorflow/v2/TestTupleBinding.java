package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_FLOAT32;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of how a tensor's type survives a binding form rather than an operation: tuple
 * destructuring, and the names the targets take. Nothing here is about what a TensorFlow API
 * computes, which is why it is not in a feature-area class for one.
 */
public class TestTupleBinding extends AbstractTensorTest {

  /**
   * A destructuring assignment whose left-hand side does not mention the name being destructured.
   * Both fields keep their types, which is the control for {@link #testRebindingDestructure}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPlainDestructureSecondField()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_rebinding_destructure.py",
        "consume_plain_second",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_5_FLOAT32)));
  }

  /**
   * The same destructuring, except that field 0's target rebinds the very name on the right, which
   * is {@code gpt-2-tensorflow2.0}'s {@code train_dataset, test_dataset = train_dataset}. Field 1
   * used to lose its type, because the read that followed the rebinding was re-pointed at field 0's
   * result rather than at the tuple (wala/ML#819).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRebindingDestructure()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_rebinding_destructure.py",
        "consume_rebound_second",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_5_FLOAT32)));
  }
}
