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
   * A batch of the zip batches each member: the first. A partial batch of unknown size is read
   * beside the full one, though every batch is full.
   *
   * <p>TODO: wala/ML#1203: expect the full batch alone.
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
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * The second batched member. A partial batch of unknown size is read beside the full one, though
   * every batch is full.
   *
   * <p>TODO: wala/ML#1203: expect the full batch alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatchedB() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_batched_b",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(INT_32, 2, 3),
                new TensorType(INT_32, List.of(new SymbolicDim("?"), new NumericDim(3))))));
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
   * The inner tuple's second dataset after {@code batch(2)}, with the partial batch of unknown size
   * every zip's batch reads.
   *
   * <p>TODO: wala/ML#1203: expect the full batch alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGroupedBatchedB() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_grouped_batched_b",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(INT_32, 2, 3),
                new TensorType(INT_32, List.of(new SymbolicDim("?"), new NumericDim(3))))));
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
}
