package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code fit}, {@code evaluate} and {@code predict} on a {@code tf.data.Dataset} give the model's
 * step the dataset's element, so the model's {@code call} receives the element's inputs: each
 * member of a structured component with its own shape and dtype, and never the targets (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>).
 */
public class TestFitDatasetElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_dataset_element.py";

  /**
   * A component that is a pair is no tensor (<a
   * href="https://github.com/wala/ML/issues/1138">wala/ML#1138</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMixedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_inputs", 0, 0);
  }

  /**
   * The float member of a pair whose members differ in dtype keeps its own dtype (<a
   * href="https://github.com/wala/ML/issues/1138">wala/ML#1138</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMixedX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The int member of the same pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMixedIds() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_ids", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /**
   * The first member of a pair of float tensors of different shapes keeps its own shape (<a
   * href="https://github.com/wala/ML/issues/1138">wala/ML#1138</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSameX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_same_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The second member of the same pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSameMask() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_same_mask", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Five rows batched by two: both batch sizes.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testShort() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_short",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 1, 4))));
  }

  /**
   * One of two models fit on datasets whose elements disagree reads only its own.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The other of the two models.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_second", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3, 6))));
  }

  /**
   * {@code predict} on a dataset of inputs alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPredicted() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_predicted", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5))));
  }

  /**
   * A member of a dict component, through {@code evaluate}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testDictA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dict_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The other member of the dict component.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testDictB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dict_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 2))));
  }

  /**
   * A {@code padded_batch} of fixed-length rows.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPadded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_padded", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * A value read of the inputs sees the inputs, not the targets (<a
   * href="https://github.com/wala/ML/issues/1141">wala/ML#1141</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testScaled() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_scaled", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6))));
  }
}
