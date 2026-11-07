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
 * The transformation {@code group_by_window} gives its dataset an element, and {@code
 * tf.data.experimental.group_by_window} and {@code bucket_by_sequence_length} handed to {@code
 * apply} give the element the methods give (<a
 * href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetGroupByWindow extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_group_by_window.py";

  /** A batch of rows of shape {@code (4,)} whose size the analysis does not count. */
  private static final TensorType BATCH =
      new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4)));

  /** Batches of two rows of shape {@code (4,)}, and a last batch of any size. */
  private static final Set<TensorType> GROUPED_ROWS = Set.of(TensorType.of(FLOAT_32, 2, 4), BATCH);

  /**
   * The element of {@code group_by_window} is the element of the datasets its {@code reduce_func}
   * returns, here a batch of two of a window. A window of three ends with a batch of one, and how
   * many elements a window holds is not counted, so the last batch's size is unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGrouped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped", 1, 1, Map.of(2, GROUPED_ROWS));
  }

  /**
   * A {@code window_size_func} in place of a {@code window_size} gives the same element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedWindowSizeFunc()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_window_size_func", 1, 1, Map.of(2, GROUPED_ROWS));
  }

  /**
   * {@code tf.data.experimental.group_by_window} handed to {@code apply} gives the element the
   * method gives.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedApplied() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_applied", 1, 1, Map.of(2, GROUPED_ROWS));
  }

  /**
   * A member of a pair element is batched in each window, as the whole element is.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedPairX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_pair_x", 1, 1, Map.of(2, GROUPED_ROWS));
  }

  /**
   * The pair's other member is batched in each window too.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedPairY() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_grouped_pair_y",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(INT_32, 2, 3),
                new TensorType(INT_32, List.of(new SymbolicDim("?"), new NumericDim(3))))));
  }

  /**
   * {@code reduce_func} is given the key {@code key_func} returns for the window's elements.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_key", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64))));
  }

  /**
   * {@code tf.data.experimental.bucket_by_sequence_length} handed to {@code apply} gives the
   * element the method gives: a batch of the receiver's element whose size is not counted. Its
   * {@code padded_shapes}, which the program does not supply, is not taken for supplied though the
   * transformation passes it on.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedApplied() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bucketed_applied", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  /**
   * A {@code padded_shapes} the program supplies to {@code
   * tf.data.experimental.bucket_by_sequence_length} may change the padded element's shape, which is
   * then unknown, as for the method. It has a fixture of its own: the dataset the transformation's
   * summary makes is one object for every {@code apply} of such a transformation, so in one program
   * a supplied {@code padded_shapes} leaves every such element's shape unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedAppliedPadded()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dataset_bucket_applied_padded.py",
        "consume_bucketed_applied_padded",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }
}
