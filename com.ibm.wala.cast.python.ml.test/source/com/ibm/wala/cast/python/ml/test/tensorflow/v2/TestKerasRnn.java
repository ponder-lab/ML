package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for the generic {@code tf.keras.layers.RNN} over a user cell (wala/ML#973): the layer
 * invokes its cell on one step of the input and builds its output from the cell's step output.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestKerasRnn extends AbstractTensorTest {

  private static final String FILE = "tf2_test_keras_rnn.py";

  /**
   * The cell's step input is the layer's input without its time axis, {@code (3, 5)} float32, the
   * same as when the program calls the cell directly on {@code x[:, 0]}. A step tensor that kept
   * the time axis would add a {@code (3, 7, 5)} member here.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCellStepInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_step_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 5))));
  }

  /**
   * A direct call on the cell types its step output, {@code (3, 5)} int32: the control showing the
   * cell dispatches without a model of its base class.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDirectCellCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_direct", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3, 5))));
  }

  /**
   * With {@code return_sequences=True}, the layer stacks one step output per time step: the cell's
   * {@code (3, 5)} int32 step output becomes {@code (3, 7, 5)} int32, the time axis taken from the
   * layer's input. Before the model the layer's call had no target and this read no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSequenceOutput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_sequence", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3, 7, 5))));
  }

  /**
   * The state the layer returns beside its output is the cell's new state, {@code (3, 5)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReturnedState()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_last_state", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 5))));
  }

  /**
   * Without {@code return_sequences} (Keras's default, {@code False}) the output is the last
   * step's, so the step shape stands: {@code (3, 5)} int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLastStepOutput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_last_output", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3, 5))));
  }
}
