package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The element of {@code flat_map} and {@code interleave} is any element of the datasets their
 * function returns, the function called on the receiver's element (<a
 * href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetFlatMapElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_flat_map_element.py";

  /** A batch of a dataset of unknown cardinality, full or short. */
  private static final Set<TensorType> BATCHES_OF_3_4 =
      Set.of(
          TensorType.of(FLOAT_32, 3, 4),
          new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))));

  /**
   * The element of a {@code flat_map} whose function returns a dataset of its argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFlatMapped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_mapped", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * The function {@code flat_map} is given is called on the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFunctionArgument() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_function_argument", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * The element of an {@code interleave} whose function returns a dataset of its argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testInterleaved() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_interleaved", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * The first member of a pair a {@code flat_map}'s function returns, the function given the
   * receiver's pair as two arguments.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPairX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_mapped_pair_x", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * The second member of that pair keeps its own type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPairY() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_mapped_pair_y", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * That pair, iterated whole, is no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_mapped_pair", 0, 0);
  }

  /**
   * A {@code flat_map} whose function slices a batch hands on the batch's rows.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnbatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unbatched", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * A function that may return either of two datasets gives either one's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testEither() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_either",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TensorType.of(FLOAT_32, 5))));
  }

  /**
   * A batch of a {@code flat_map}, whose cardinality is unknown, may be short.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFlatMappedBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_mapped_batched", 1, 1, Map.of(2, BATCHES_OF_3_4));
  }

  /**
   * So may a batch of an {@code interleave}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testInterleavedBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_interleaved_batched", 1, 1, Map.of(2, BATCHES_OF_3_4));
  }

  /**
   * The lines of the text files an {@code interleave} reads, one {@code TextLineDataset} per file.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testLines() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_lines", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * The {@code x} a model's {@code call} gets from {@code fit} over a batched {@code flat_map} of
   * pairs is the batched first member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFitX() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_fit_x",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * A tuple element longer than the four positions the analysis spreads leaves the function
   * uncalled, so the element is unknown, as {@code map}'s is, never the dataset of the default the
   * function returns when it is called with fewer.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFifth() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fifth", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * A function that returns no dataset gives no element: the tensor it returns is not read as one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNotADataset() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_not_a_dataset", 0, 0);
  }

  /**
   * The {@code window(n).flat_map(lambda w: w.batch(n))} idiom reads each window's batch: a full
   * batch, or a short one, since a window's size is unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testWindowedBatch() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_windowed_batch",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }
}
