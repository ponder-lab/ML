package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests Keras learning-rate schedules, which an optimizer calls with its step count cast to the
 * variables' dtype when it applies gradients (<a
 * href="https://github.com/wala/ML/issues/1050">wala/ML#1050</a>).
 */
public class TestLearningRateSchedule extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_lr_schedule.py";

  /**
   * A schedule given to an {@code Adam} that applies gradients, and called directly with an {@code
   * int64} step: {@code step} holds both. Before this, only the direct call reached it.
   */
  @Test
  public void testApplyGradientsCallsSchedule()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "Decay.__call__",
        1,
        8,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT64, SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** {@code minimize} calls the schedule as {@code apply_gradients} does. */
  @Test
  public void testMinimizeCallsSchedule()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "Minimized.__call__", 1, 3, Map.of(3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** TensorFlow Addons' {@code LazyAdam} is an {@code Adam}, and calls its schedule as one. */
  @Test
  public void testLazyAdamCallsSchedule()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "Lazy.__call__", 1, 3, Map.of(3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * A schedule no optimizer is given is called only by the program, so its {@code step} gains no
   * {@code float32} member.
   */
  @Test
  public void testScheduleWithoutOptimizerKeepsItsSteps()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "DirectOnly.__call__", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT64)));
  }
}
