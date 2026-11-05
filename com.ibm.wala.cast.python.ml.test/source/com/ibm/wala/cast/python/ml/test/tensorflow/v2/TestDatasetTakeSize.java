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
 * The dataset of {@code take(count)} has the least of {@code count} and its receiver's size, so a
 * batch of it counts the elements it keeps (<a
 * href="https://github.com/wala/ML/issues/1216">wala/ML#1216</a>).
 */
public class TestDatasetTakeSize extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_take_size.py";

  /** A full batch of 2 or a short one of 1. */
  private static final Set<TensorType> BATCHES_OF_2_OR_1 =
      Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 1, 4));

  /**
   * Three elements batch by 2 into a full batch and one of 1.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTakenBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_taken_batched", 1, 1, Map.of(2, BATCHES_OF_2_OR_1));
  }

  /**
   * A count of -1 takes every element: 8 batch by 3 into two full batches and one of 2.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAllTakenBatched() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_all_taken_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 4), TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A count past the receiver's size takes every element: 8 batch by 4 into two full batches.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testOverTakenBatched() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE, "consume_over_taken_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 4))));
  }

  /**
   * A count that does not resolve leaves the size unknown, so a batch may be short.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnknownTakeBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unknown_take_batched",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * A member of a pair counts the same elements.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTakenPairX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_taken_pair_x", 1, 1, Map.of(2, BATCHES_OF_2_OR_1));
  }

  /**
   * A take of a repeated dataset is finite and has exactly {@code count} elements: 5 batch by 2
   * into two full batches and one of 1.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRepeatedTakenBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_repeated_taken_batched", 1, 1, Map.of(2, BATCHES_OF_2_OR_1));
  }
}
