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
 * The transformations {@code skip}, {@code shard} and {@code cache} hand on their receiver's
 * element, and a batch of one counts the elements it keeps (<a
 * href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetPassThroughMethods extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_pass_through_methods.py";

  /**
   * The element of {@code skip} is the receiver's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSkipped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_skipped", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The element of {@code shard} is the receiver's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSharded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sharded", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The element of {@code cache} is the receiver's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testCached() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_cached", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A member of the receiver's pair keeps its own type after {@code skip}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSkippedPairX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_skipped_pair_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The pair's other member keeps its own type too.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSkippedPairY() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_skipped_pair_y", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A batch by 2 of 8 elements less 1: three full batches and one of 1.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSkippedBatched() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_skipped_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 1, 4))));
  }

  /**
   * A batch by 2 of every second of 8 elements, 4, every batch full.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testShardedBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sharded_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A batch by 2 of every third of 8 elements from the second, 3: one full batch and one of 1.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnevenShardBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_uneven_shard_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 1, 4))));
  }

  /**
   * A batch by 2 of a cache of 8 elements, every batch full.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testCachedBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_cached_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A pass-through before {@code skip(2)} keeps the count: a batch by 2 of 6 elements, every batch
   * full.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testChainedBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_chained_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A count the analysis cannot resolve leaves the size unknown, so a partial batch of unknown size
   * is read beside the full one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnknownSkipBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unknown_skip_batched",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }
}
