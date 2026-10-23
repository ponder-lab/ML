package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for the built-in Keras recurrent cells, {@code SimpleRNNCell}, {@code GRUCell}, {@code
 * LSTMCell}, and TensorFlow Addons' {@code PeepholeLSTMCell}: a call on one is a step of width
 * {@code units} in the float32 compute dtype, and a generic {@code tf.keras.layers.RNN} over one
 * builds its output from that step.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestRnnBuiltinCells extends AbstractTensorTest {

  private static final String FILE = "tf2_test_rnn_builtin_cells.py";

  /**
   * An {@code RNN} over a {@code SimpleRNNCell}, called with keyword {@code inputs} and {@code
   * initial_state}, outputs {@code (2, 3, 4)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSimpleRnnCell()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_simple", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * As {@link #testSimpleRnnCell()}, over a {@code GRUCell}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGruCell()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_gru", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * As {@link #testSimpleRnnCell()}, over an {@code LSTMCell} with a two-tensor initial state.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLstmCell()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_lstm", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * As {@link #testLstmCell()}, over TensorFlow Addons' {@code PeepholeLSTMCell}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPeepholeLstmCell()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_peephole", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * A direct call on a {@code SimpleRNNCell} with one step, {@code (2, 5)}, outputs {@code (2, 4)}
   * float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDirectCellOutput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_cell_output", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * The new state a direct {@code SimpleRNNCell} call returns is its output, {@code (2, 4)}
   * float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDirectCellState()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_cell_state", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * An {@code LSTMCell}'s new carry state, the second of its two states, is {@code (2, 4)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLstmCellCarry()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_lstm_cell_carry", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * An {@code RNN} over a subclass of a built-in cell that overrides {@code call} takes its step
   * from the override, {@code (2, 3, 1)} int32, not from the built-in cell's call.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubclassOverrideSequence()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE, "consume_subclass_sequence", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3, 1))));
  }

  /**
   * A direct call on a subclass of a built-in cell that overrides {@code call} returns the
   * override's output, {@code (2, 1)} int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubclassOverrideDirect()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_subclass_direct", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 1))));
  }
}
