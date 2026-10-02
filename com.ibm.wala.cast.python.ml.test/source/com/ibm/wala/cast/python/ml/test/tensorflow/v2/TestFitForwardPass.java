package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a model trained only through {@code fit} runs its forward pass (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): through the default {@code
 * train_step}, which calls the model on the batch's inputs, or through the model's own {@code
 * train_step} override, which receives the {@code (x, y)} batch.
 */
public class TestFitForwardPass extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_forward_pass.py";

  private static final TensorType X = TensorType.of(FLOAT_32, 8, 4);

  private static final TensorType Y = TensorType.of(FLOAT_32, 8, 3);

  /** The layer's {@code call} is reached from {@code fit} with the training inputs. */
  @Test
  public void testLayerCallReached() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inner", 1, 1, Map.of(2, Set.of(X)));
  }

  /**
   * The layer's build-time constraint receives the weight, through the forward pass {@code fit}
   * runs.
   */
  @Test
  public void testConstraintReached() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "FitNorm.__call__", 1, 2, Map.of(3, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /** A {@code train_step} override is reached from {@code fit} and unpacks the batch's inputs. */
  @Test
  public void testTrainStepInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_step_x", 1, 1, Map.of(2, Set.of(X)));
  }

  /** A {@code train_step} override unpacks the batch's targets. */
  @Test
  public void testTrainStepTargets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_step_y", 1, 1, Map.of(2, Set.of(Y)));
  }

  /** {@code predict} returns the forward pass's output. */
  @Test
  public void testPrediction() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_prediction", 1, 1, Map.of(2, Set.of(Y)));
  }
}
