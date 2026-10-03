package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Iterating a dataset yields its elements, not the dataset (<a
 * href="https://github.com/wala/ML/issues/1010">wala/ML#1010</a>). The iterator's {@code __next__}
 * allocates the element with a component at each constant index, so the loop variable, its unpacked
 * components, a dict-keyed component, and the elements {@code enumerate}, {@code zip} and {@code
 * next(iter(...))} pass through carry the element's shape and dtype.
 */
public class TestDatasetIteration extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_dataset_iteration.py";

  private static final TensorType BATCH = TensorType.of(FLOAT_32, 8, 4);

  @Test
  public void testUnpackedFirstComponent()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_x", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  @Test
  public void testUnpackedSecondComponent()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_y", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 8))));
  }

  @Test
  public void testWholeElement() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_single", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  @Test
  public void testKeyedComponent() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_keyed", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  @Test
  public void testEnumeratedElement() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_enumerated", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  @Test
  public void testZippedElements() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_zipped", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  @Test
  public void testNextOfIter() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_next", 1, 1, Map.of(2, Set.of(BATCH)));
  }

  /** A constant index into a single-tensor element is a tensor subscript, not a component. */
  @Test
  public void testRowOfSingleTensorElement()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A loop over either of two datasets binds an element of each, so the loop variable is typed by
   * both datasets' elements.
   */
  @Test
  public void testElementOfEitherDataset()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_either", 1, 1, Map.of(2, Set.of(BATCH, TensorType.of(FLOAT_32, 8, 3))));
  }

  /** An unpacked component of a loop over either of two pair datasets is typed by both. */
  @Test
  public void testComponentOfEitherDataset()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE, "consume_either_x", 1, 1, Map.of(2, Set.of(BATCH, TensorType.of(FLOAT_32, 8, 5))));
  }
}
