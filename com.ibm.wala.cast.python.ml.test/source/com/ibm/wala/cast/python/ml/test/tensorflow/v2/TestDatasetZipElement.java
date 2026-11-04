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
 * {@code zip} yields the structure it is given with each dataset replaced by that dataset's
 * element, so each member keeps its own type (<a
 * href="https://github.com/wala/ML/issues/164">wala/ML#164</a>).
 */
public class TestDatasetZipElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_zip_element.py";

  /**
   * A zipped pair is a tuple, no tensor.
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
   * The first member keeps the first dataset's type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The second member keeps the second dataset's type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A batch of the zip batches each member: the first. Every batch is full, since the zip has as
   * many elements as its shortest dataset, 8, which 2 divides (wala/ML#1203).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatchedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_batched_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The second batched member, every batch of which is full (wala/ML#1203).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatchedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_batched_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /**
   * A zipped dataset whose element is a pair keeps the pair's members: the first.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNestedX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_nested_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * The pair's second member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNestedMask() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_nested_mask", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A zip of a dict of datasets is a dict of their elements: the first entry.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The second entry.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A zip of a mapped dataset, whose element is not copied, keeps the element its generator types:
   * the mapped member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMappedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mapped_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The other member of that zip.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMappedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mapped_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A zip of a nested tuple of datasets nests their elements: the first dataset of the inner tuple.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The second dataset of the inner tuple.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * The dataset beside the inner tuple.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedC() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_c", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * The inner tuple's second dataset after {@code batch(2)}, every batch of which is full, since
   * the zip has as many elements as its datasets, 8 (wala/ML#1203).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedBatchedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_batched_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /**
   * An entry of a dict of datasets nested in the zipped tuple.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedKeyedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grouped_keyed_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A zip of a namedtuple of datasets nested in the zipped tuple: the namedtuple's first member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNamedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_named_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The namedtuple's second member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNamedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_named_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * The namedtuple's second member read by its name, which the element's copy holds as it holds the
   * member's position (https://github.com/wala/ML/issues/1210).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNamedFieldB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_named_field_b", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A batch of a zip of datasets of different sizes: the zip has as many elements as the shorter,
   * 5, so its batches of 2 end with one of 1 (wala/ML#1203).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSizedUnevenA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_sized_uneven_a",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_32, 1, 4))));
  }

  /**
   * A member of a batch of a zip of a dict of datasets, every batch of which is full
   * (wala/ML#1203).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSizedKeyedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sized_keyed_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A batch of a zip with a dataset of no known size: the zip's size is unknown too, so a partial
   * batch of unknown size is read beside the full one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSizedUnknownA() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_sized_unknown_a",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 3, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * A batch of a zip of a namedtuple of datasets with another: every batch is full, since a
   * namedtuple's datasets are zipped as a tuple's are (wala/ML#1203).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSizedNamedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sized_named_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
