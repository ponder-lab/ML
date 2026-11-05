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
 * {@code window} yields datasets whose own iteration yields the source elements, and for a tuple
 * element a tuple of datasets, one per component (<a
 * href="https://github.com/wala/ML/issues/1201">wala/ML#1201</a>).
 */
public class TestDatasetWindowElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_window_element.py";

  /**
   * A window is a dataset, which, as any dataset value, reads as a tensor of its element's type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testWindow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_window", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
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
   * The first of a pair's windows is a dataset of the pair's first members, which reads as a tensor
   * of their type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFirstWindow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first_window", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
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

  /**
   * A window has the dataset methods, and its size is not the receiver's: a window may hold fewer
   * elements than its size, so a batch of it may be short.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testWindowBatch() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_window_batch",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * A row of a window of a dataset whose element is not mirrored, a batch of a {@code
   * TextLineDataset}'s lines, is read through the window's own generator, which reads the
   * receiver's element: a batch of the lines, full or short, not a row of one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testLines() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_lines",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(STRING, List.of(new NumericDim(2))),
                new TensorType(STRING, List.of(new SymbolicDim("?"))))));
  }

  /**
   * The rows of the window of the fifth component of a five-component element, turned into arrays
   * and concatenated, keep the declared dtype (<a
   * href="https://github.com/wala/ML/issues/1221">wala/ML#1221</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFifthRows() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fifth_rows", 1, 1, Map.of(2, Set.of(new TensorType(INT_32, null))));
  }

  /**
   * A row of the window of position 15, the last position the analysis makes a window of, has its
   * component's type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSixteenthRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sixteenth_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 16))));
  }

  /**
   * Position 16 has no window in the analysis, so it reads as no tensor: the first position past
   * the bound (<a href="https://github.com/wala/ML/issues/1221">wala/ML#1221</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSeventeenthWindow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_seventeenth_window", 0, 0);
  }
}
