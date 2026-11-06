package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code scan} calls its function with the state and the element, and yields the second member of
 * each pair the function returns, threading the first as the next state (<a
 * href="https://github.com/wala/ML/issues/1214">wala/ML#1214</a>).
 */
public class TestDatasetScanElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_scan_element.py";

  /**
   * The function is given the initial state and the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testStepArguments() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "step",
        2,
        3,
        Map.of(2, Set.of(TensorType.of(FLOAT_32)), 3, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The element is the second member of the function's return, here its element argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testStepped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stepped", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A state whose shape the function changes is given as both the initial state and each new state.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGrowArguments() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "grow",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32), TensorType.of(FLOAT_32, 4)),
            3,
            Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * An element that is the state the function was given has every shape the state may have.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGrown() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_grown",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32), TensorType.of(FLOAT_32, 4))));
  }

  /**
   * An element that is the state, not the receiver's element, has the state's type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testHeld() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_held", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /**
   * A batch of the scanned dataset batches the function's return, not the receiver's element, and
   * keeps the receiver's size, so its last batch is partial.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testHeldBatch() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_held_batch",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3), TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A batch of an element the function computes has the computed type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTotalBatch() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_total_batch", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A map of the scanned dataset is given the function's return.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testHeldMapped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_held_mapped", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /**
   * A member of a structured second member, such as a tuple, is that tuple's member, which the
   * function returns.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSplitFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_split_first", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A batch of a structured second member is not modeled: the generator does not read a member of a
   * structure, so a member of the batch is no tensor rather than a tensor of unknown type standing
   * in for the missing model (wala/ML#1214).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSplitBatchFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_split_batch_first", 0, 0, Map.of());
  }
}
