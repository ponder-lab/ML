package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code fit}, {@code evaluate} and {@code predict} call the hooks of the Keras
 * callbacks they are passed (<a href="https://github.com/wala/ML/issues/1244">wala/ML#1244</a>), so
 * a hook and what it calls are in the call graph. Each hook calls a function of its own, which has
 * a call-graph node only if the hook is called.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestFitCallbacks extends AbstractTensorTest {

  private static final String FILE = "tf2_test_keras_callbacks.py";

  /**
   * {@code fit} calls a callback's {@code on_epoch_end}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEpochEnd() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "reached_epoch_end", 0, 0);
  }

  /**
   * {@code fit} calls a callback's {@code on_train_batch_end}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTrainBatchEnd() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "reached_train_batch_end", 0, 0);
  }

  /**
   * {@code evaluate} calls a callback's {@code on_test_end}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTestEnd() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "reached_test_end", 0, 0);
  }

  /**
   * {@code predict} calls a callback's {@code on_predict_end}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPredictEnd() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "reached_predict_end", 0, 0);
  }

  /**
   * A hook's {@code self.model} is the model being trained, as Keras's {@code set_model} sets it,
   * so calling it gives the model's output.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testHookModel() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "model_output", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  /**
   * A callback appended to the list {@code fit} is passed has its hooks called too.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAppendedCallback() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_keras_callbacks_appended.py", "reached_epoch_end", 0, 0);
  }
}
