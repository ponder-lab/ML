package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a training loop reached through a generator of steps (<a
 * href="https://github.com/wala/ML/issues/1051">wala/ML#1051</a>): an instance call iterates a
 * generator method, which wraps a nested closure in {@code tf.function} and calls it, and the
 * closure calls the instance's step method. Each test pins one link of the chain.
 */
public class TestTrainerSteps extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_trainer_steps.py";

  /** The instance call. */
  @Test
  public void testCall() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_call", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** The generator method the call iterates. */
  @Test
  public void testSteps() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_steps", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** The nested closure the generator wraps in {@code tf.function} and calls. */
  @Test
  public void testClosure() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_closure", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** The gradients the step method hands to the accumulator. */
  @Test
  public void testAccumulator() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_forward", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }
}
