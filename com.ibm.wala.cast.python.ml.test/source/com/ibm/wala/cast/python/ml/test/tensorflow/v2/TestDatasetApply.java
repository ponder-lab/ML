package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The dataset {@code Dataset.apply} hands the function it applies is typed in that function as the
 * receiver is, and so is what {@code apply} returns for a function that hands the dataset back (<a
 * href="https://github.com/wala/ML/issues/1290">wala/ML#1290</a>).
 */
public class TestDatasetApply extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_apply.py";

  private static final String LOOP_FILE = "tf2_test_dataset_apply_loop.py";

  /**
   * What a dataset reassigned in a loop to what {@code apply} returns for it reads: the first
   * dataset's rows and a batch of them by each of the two functions, without the batch of batches
   * the second function returns.
   */
  private static final Set<TensorType> LOOPED =
      Set.of(
          TensorType.of(FLOAT_32, 23),
          TensorType.of(FLOAT_32, 2, 23),
          TensorType.of(FLOAT_32, 3, 23));

  /**
   * What the parameter of a function applied in such a loop reads: the first dataset's rows, and
   * the unknown shape for the datasets the loop hands back to it.
   */
  private static final Set<TensorType> LOOP_FED =
      Set.of(TensorType.of(FLOAT_32, 23), TENSOR_UNKNOWN_SHAPE_FLOAT32);

  /**
   * The parameter of an applied function reads the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testParameter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "hand_back", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * What {@code apply} returns for a function that hands its parameter back reads the receiver's
   * element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testHandedBack() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "hand_back_other", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
    test(FILE, "consume_handed_back", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * A function that batches its parameter reads the rows in the parameter, and what {@code apply}
   * returns for it reads the batch.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "batch_rows", 1, 2, Map.of(2, Set.of(TENSOR_7_FLOAT32)));
    test(FILE, "consume_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 7))));
  }

  /**
   * A dataset whose element is not mirrored, here a file's lines, is typed in the applied function
   * and where {@code apply} returns it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testLines() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "hand_back_lines", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    test(FILE, "consume_lines", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * A function called from the lambda that is applied reads the receiver's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testInLambda() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_lambda", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 11))));
  }

  /**
   * A function the applied lambda passes its dataset to, for the shapes of the dataset's specs,
   * reads the receiver's element, and the batch padded to those shapes reads as before.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testShapesOfParameter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "shapes_of", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 13))));
    test(FILE, "consume_padded", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 13))));
  }

  /**
   * A function applied to a window inside a {@code reduce_func} reads the window's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedToWindow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "batch_window", 1, 2, Map.of(2, Set.of(TensorType.of(FLOAT_32, 19))));
  }

  /**
   * Of two functions applied one after the other, the second reads what the first returns.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testChained() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "chained_first", 1, 2, Map.of(2, Set.of(TensorType.of(FLOAT_32, 29))));
    test(FILE, "chained_second", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 29))));
  }

  /**
   * A function that is both called with a dataset and applied to another reads each dataset's
   * element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testCalledAndApplied() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "called_and_applied",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 31), TensorType.of(FLOAT_32, 37))));
  }

  /**
   * A dataset of dicts whose element is not mirrored reads a tensor of unknown shape and dtype in
   * the applied function, as it does in a function it is passed to.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testDictElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "pass_features", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
    test(FILE, "hand_back_features", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A row read by iterating the parameter in the applied function reads the receiver's element, as
   * before.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testIterated() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "iterate", 1, 3, Map.of(2, Set.of(TensorType.of(FLOAT_32, 43))));
    test(FILE, "consume_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 43))));
  }

  /**
   * A control: the parameter of a function the dataset is passed to reads the dataset's element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPassed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "batch_directly", 1, 3, Map.of(2, Set.of(TensorType.of(FLOAT_32, 47))));
  }

  /**
   * The rows of a {@code tf.data.Dataset.zip} of the parameter, read in the applied function, read
   * no tensor.
   *
   * <p>TODO: wala/ML#1290. Once fixed, each row is {@code (53,)} of {@code float32}, as a row of a
   * zip of a dataset passed to a function is.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZipOfParameter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "zip_parameter", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 53))));
    test(FILE, "consume_zipped", 0, 0, Map.of());
    test(
        FILE,
        "consume_zipped_passed",
        2,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 59)), 3, Set.of(TensorType.of(FLOAT_32, 59))));
  }

  /**
   * The parameters of two functions applied in a loop that reassigns the dataset read the join of
   * the datasets the loop variable holds.
   *
   * <p>Each function batches what the other returned, so the datasets' shapes form a cycle that
   * adds an axis on each turn. No shape stands for a rank that grows without end, so the datasets
   * the loop hands back read as the unknown shape, which covers the {@code (2, 23)} that {@code
   * second} is given at run time.
   *
   * <p>TODO: wala/ML#1197. Once fixed, the read may name the batches the loop makes, {@code (2,
   * 23)} and {@code (3, 2, 23)}, where it holds the unknown shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAppliedInLoop() throws ClassHierarchyException, CancelException, IOException {
    test(LOOP_FILE, "first", 1, 2, Map.of(2, LOOP_FED));
    test(LOOP_FILE, "second", 1, 2, Map.of(2, LOOP_FED));
  }

  /**
   * The dataset a loop of {@code apply} calls leaves reads no batch of batches.
   *
   * <p>TODO: wala/ML#1197. Once fixed, the types include {@code (3, 2, 23)}, which the dataset is
   * at run time, or a shape whose leading axes are unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testLoopResult() throws ClassHierarchyException, CancelException, IOException {
    test(LOOP_FILE, "consume_looped", 1, 1, Map.of(2, LOOPED));
  }
}
