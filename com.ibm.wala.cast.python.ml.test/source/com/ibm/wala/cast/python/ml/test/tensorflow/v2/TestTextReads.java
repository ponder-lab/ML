package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for text reads feeding a dataset: a file's lines, sliced into a dataset and mapped, give
 * the map callback a scalar string element.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestTextReads extends AbstractTensorTest {

  /**
   * {@code open(path).read().splitlines()} is a list of strings, {@code from_tensor_slices} over it
   * a dataset of scalar strings, and the map callback's {@code line} one of them. Before, the
   * top-level {@code open} resolved through an import of a module named {@code open} to nothing,
   * and neither the file's {@code read} nor the string's {@code splitlines} had a model, so {@code
   * line} read an unknown dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLineFromFileLines()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_records.py",
        "consume_line",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * A file's {@code readlines()} is a list of strings too, and a dataset sliced from it yields
   * scalar strings.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testElementFromReadlines()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_records.py",
        "consume_readline_element",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * A string's {@code split(...)} is a list of strings, and a dataset sliced from it yields scalar
   * strings.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPieceFromSplit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_records.py",
        "consume_split_piece",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }
}
