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
 * The elements of the transformations {@code unbatch}, {@code rejection_resample} and {@code
 * bucket_by_sequence_length} read a member at a time, and read past a further transformation (<a
 * href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetElementTransformationReads extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_element_transformation_reads.py";

  /**
   * A member of a bucketed pair is a batch of that member, of a size the analysis does not count.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedPairX() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_bucketed_pair_x",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * The pair's other member is a batch of its own member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedPairY() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_bucketed_pair_y",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(INT_32, List.of(new SymbolicDim("?"), new NumericDim(3))))));
  }

  /**
   * A batch of a bucketed dataset stacks its batches, and since the number of batches isn't
   * counted, the last may be partial.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedBatch() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_bucketed_batch",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32, List.of(new NumericDim(2), new SymbolicDim("?"), new NumericDim(4))),
                new TensorType(
                    FLOAT_32,
                    List.of(new SymbolicDim("?"), new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * The class of a resampled element, read by its index, is an int32 scalar.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testResampledClassRead()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_resampled_class_read", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * A member of a resampled element read past a {@code skip} is the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testResampledSkipped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_resampled_skipped", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A member of an unbatched dictionary element, read by its key, is a row of that member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnbatchedDict() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unbatched_dict", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The class of a resampled element of a dataset made from a list of pairs, read by its index, is
   * an int32 scalar.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testResampledListClass()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_resampled_list_class", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * A member of a bucketed pair a map returns is a batch of that member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBucketedMappedX() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_bucketed_mapped_x",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }
}
