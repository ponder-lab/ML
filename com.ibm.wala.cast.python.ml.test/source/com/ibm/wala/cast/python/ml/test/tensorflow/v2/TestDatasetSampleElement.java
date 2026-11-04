package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The element of {@code sample_from_datasets} and {@code choose_from_datasets} is any one of the
 * elements of the datasets they are given (<a
 * href="https://github.com/wala/ML/issues/1215">wala/ML#1215</a>), so a tuple element is no tensor,
 * and each member keeps its own type.
 */
public class TestDatasetSampleElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_sample_element.py";

  /**
   * The element of {@code sample_from_datasets} over pair datasets is a pair, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSampledPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sampled_pair", 0, 0);
  }

  /**
   * The first member of a sampled element is either dataset's, each shape with its own dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSampledA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_sampled_a",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TensorType.of(FLOAT_32, 5))));
  }

  /**
   * The second member of a sampled element is either dataset's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSampledB() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_sampled_b",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32, TensorType.of(INT_32, 2))));
  }

  /**
   * A batch of a sampled dataset batches either dataset's member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSampledBatchedA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_sampled_batched_a",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 4), TensorType.of(FLOAT_32, 1, 5))));
  }

  /**
   * The element of {@code choose_from_datasets} over pair datasets is a pair, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testChosenPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_chosen_pair", 0, 0);
  }

  /**
   * The first member of a chosen element is either dataset's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testChosenA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_chosen_a",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TensorType.of(FLOAT_32, 5))));
  }

  /**
   * The second member of a chosen element is either dataset's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testChosenB() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_chosen_b",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32, TensorType.of(INT_32, 2))));
  }

  /**
   * The inputs a model's {@code call} gets from a sample of datasets whose element is {@code ((x,
   * mask), y)} are the pair {@code (x, mask)}, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFitInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fit_inputs", 0, 0);
  }

  /**
   * The {@code x} of those inputs is the batched {@code x}, without {@code mask} or {@code y}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFitX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fit_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
