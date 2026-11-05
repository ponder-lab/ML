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
 * The transformations {@code unbatch}, {@code apply}, {@code rejection_resample} and {@code
 * bucket_by_sequence_length} give their datasets an element (<a
 * href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetElementTransformations extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_element_transformations.py";

  /** A batch of rows of shape {@code (4,)} whose size the analysis does not count. */
  private static final TensorType BATCH =
      new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4)));

  /**
   * The element of {@code unbatch} is a row of the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnbatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unbatched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A member of the receiver's pair is a row of the batched member after {@code unbatch}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnbatchedPairX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unbatched_pair_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The pair's other member is a row of its own batched member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnbatchedPairY() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unbatched_pair_y", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * The element of {@code unbatch} over a dataset of matrices is a row of a matrix.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnbatchedRows() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unbatched_rows", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A batch after {@code unbatch} does not count the rows, so it may be partial of a size the
   * analysis does not know.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRebatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_rebatched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 4), BATCH)));
  }

  /**
   * The element of {@code apply} is the element of the dataset its function returns, here a {@code
   * skip} of its argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testApplied() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_applied", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The element of {@code apply} whose function batches its argument is the batch, which counts the
   * receiver's elements.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedBatch() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_applied_batch", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The first member of the element of {@code rejection_resample} is its class.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testResampledClass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_resampled_class", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * The second member of the element of {@code rejection_resample} is the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testResampled() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_resampled", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A member of the receiver's pair, at position 1 of the element of {@code rejection_resample},
   * keeps its own type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testResampledPairX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_resampled_pair_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The element of {@code bucket_by_sequence_length} is a batch of the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bucketed", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  /**
   * An explicit {@code pad_to_bucket_boundary=False} leaves the padded element's shape as it is.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedUnpadded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bucketed_unpadded", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  /**
   * A declared {@code padded_shapes} may change the padded element's shape, which is then unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedPadded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bucketed_padded", 1, 1, Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }
}
