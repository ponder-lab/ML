package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The element of {@code concatenate} is either dataset's element (<a
 * href="https://github.com/wala/ML/issues/1215">wala/ML#1215</a>), so a tuple element is no tensor,
 * and each member keeps its own type.
 */
public class TestDatasetConcatenateElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_concatenate_element.py";

  /**
   * The element of a concatenation of pair datasets is a pair, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair", 0, 0);
  }

  /**
   * The first member of a dataset concatenated with itself.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSameA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_same_a", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * The second member of a dataset concatenated with itself.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSameB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_same_b", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * The first member of a concatenation of two datasets is either one's, each shape with its own
   * dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testOtherA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_other_a",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TensorType.of(FLOAT_32, 5))));
  }

  /**
   * The second member of a concatenation of two datasets is either one's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testOtherB() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE, "consume_other_b", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32, TensorType.of(INT_32, 2))));
  }

  /**
   * A batch of a concatenation of two datasets batches either one's member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatchedA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_batched_a",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 2, 5))));
  }

  /**
   * The member of a {@code from_generator} dataset's tuple element, declared by {@code
   * output_types} alone, iterated on its own (<a
   * href="https://github.com/wala/ML/issues/1217">wala/ML#1217</a>): the declared dtype, and the
   * shape the generator yields.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGeneratedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_generated_a", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * A concatenation with a {@code from_generator} dataset keeps the element that dataset's
   * generator types: a member is a tensor (<a
   * href="https://github.com/wala/ML/issues/1217">wala/ML#1217</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnmodeledA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unmodeled_a", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * The inputs a model's {@code call} gets from a concatenation of datasets whose element is {@code
   * ((x, mask), y)} are the pair {@code (x, mask)}, no tensor.
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

  /**
   * A dataset of one element concatenated with itself has two, so its batch of 2 is full.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSelfBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_self_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A receiver that may be a dataset of one element or of two, concatenated with one of one, has 2
   * or 3: its batches of 2 are full, and of 3 the last is not. The sizes are summed per dataset
   * given, not over every dataset the receiver may be.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPickedBatched() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_picked_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 1, 4))));
  }
}
