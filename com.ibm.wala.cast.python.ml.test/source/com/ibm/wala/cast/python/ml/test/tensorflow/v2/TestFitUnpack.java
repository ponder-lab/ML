package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@code fit} on datasets whose steps unpack the data with {@code
 * tf.keras.utils.unpack_x_y_sample_weight} (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): a dataset of {@code (dict,
 * targets, weights)} elements yields the targets and weights as the element's components, and a
 * dataset of single tensors yields the whole element as the inputs.
 */
public class TestFitUnpack extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_unpack.py";

  /** The unpacked targets are the tuple element's second component. */
  @Test
  public void testTargets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_targets", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The unpacked sample weights are the tuple element's third component. */
  @Test
  public void testWeights() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_weights", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /** For a dataset of single tensors, the unpacked inputs are the whole element. */
  @Test
  public void testSingleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_single_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** The model's call sees exactly the single-tensor element. */
  @Test
  public void testSingleCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_single_call", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
