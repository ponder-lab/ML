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
 * The dataset of {@code filter} keeps an uncounted number of its receiver's elements, so a batch of
 * it may end with a partial batch, and a filter of a dataset that never ends never ends either (<a
 * href="https://github.com/wala/ML/issues/1237">wala/ML#1237</a>).
 */
public class TestDatasetFilterSize extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_filter_size.py";

  /**
   * A filter of four elements batches by 2 into full batches and a last batch of unknown size.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFilteredBatched() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_filtered_batched",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(INT_32, 2), new TensorType(INT_32, List.of(new SymbolicDim("?"))))));
  }

  /**
   * Without the filter, four elements batch by 2 into two full batches, as a control for {@link
   * #testFilteredBatched()}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnfilteredBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unfiltered_batched", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * A filter of an endless repeat is endless, so every batch of it is full.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRepeatedFilteredBatched()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_repeated_filtered_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * A member of a filtered pair keeps its receiver's element, and batches as the filter does.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFilteredPairX() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_filtered_pair_x",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(3))))));
  }
}
