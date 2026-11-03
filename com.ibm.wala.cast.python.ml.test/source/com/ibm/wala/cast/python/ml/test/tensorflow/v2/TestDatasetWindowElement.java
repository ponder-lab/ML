package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code window} yields datasets whose own iteration yields the source elements, and for a tuple
 * element a tuple of datasets, one per component (<a
 * href="https://github.com/wala/ML/issues/164">wala/ML#164</a>).
 */
public class TestDatasetWindowElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_window_element.py";

  /**
   * A window is a dataset, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testWindow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_window", 0, 0);
  }

  /**
   * An element of a window is a source element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The first of a pair's windows is a dataset, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFirstWindow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first_window", 0, 0);
  }

  /**
   * An element of the first window is the pair's first component.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFirstRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * An element of the second window is the pair's second component.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSecondRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_second_row", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }
}
