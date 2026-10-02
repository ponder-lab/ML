package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code fit} on a {@code tf.data.Dataset} feeds the model the dataset's element (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): the {@code train_step} override
 * unpacks the batch's inputs and targets, and the model's {@code call} sees exactly the inputs, a
 * single type and not a union with the targets or an unknown.
 */
public class TestFitDataset extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_dataset.py";

  /** The step's inputs are the element's first component. */
  @Test
  public void testStepInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_ds_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** The step's targets are the element's second component. */
  @Test
  public void testStepTargets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_ds_y", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The model's inputs are exactly the element's first component. */
  @Test
  public void testCallInputsExact() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_call_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
