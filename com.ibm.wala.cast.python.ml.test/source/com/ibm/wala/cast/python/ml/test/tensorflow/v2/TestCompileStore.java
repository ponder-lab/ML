package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code compile} stores the loss and the optimizer on the model, as Keras does, so a
 * {@code train_step} that calls {@code self.loss} and {@code self.optimizer} reaches them (<a
 * href="https://github.com/wala/ML/issues/1146">wala/ML#1146</a>). Before this, {@code compile}
 * stored nothing, and neither the loss's {@code call} nor the optimizer's schedule was reached.
 */
public class TestCompileStore extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_compile_store.py";

  /** The loss given to {@code compile} receives the labels and the model's predictions. */
  @Test
  public void testLossCalledThroughCompile()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "MyLoss.call",
        2,
        5,
        Map.of(3, Set.of(TENSOR_4_3_FLOAT32), 4, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * The optimizer given to {@code compile} applies the gradients, and calls its schedule with its
   * step count cast to {@code float32} (wala/ML#1050).
   */
  @Test
  public void testOptimizerCalledThroughCompile()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "Schedule.__call__", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }
}
