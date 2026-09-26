package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tensor readings around list repetition and concatenation (wala/ML#960). The model carries the
 * operands' ELEMENTS (the pointer-analysis witness lives in the front end's {@code
 * TestListOperations}); it never carries a length, so every reading here equals the reading without
 * the model. The three decline pins go red if any reader derives an extent from a synthesized list.
 */
public class TestListOperations extends AbstractTensorTest {

  private static final String FILE = "tf2_test_list_operations.py";

  /**
   * A layer reached through {@code [None] * n}: with the elements known, the dead {@code past}
   * arm's unstack reads its None-only input as no tensor (wala/ML#961), and the arm's concat over
   * that piece is no tensor either (wala/ML#962), so neither the unknown-dtype twin nor the
   * rankless float32 member the arm used to add beside the real {@code (2, 3, 4)} survives.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepeatedNoneKeepsTheDeadArmTwin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_repeated", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * The same layer with a direct {@code past=None}: the same reading, the two idioms reaching the
   * read with the same evidence (wala/ML#961 and wala/ML#962 read both arms the same way).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDirectNoneReadsTheSame()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_direct", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * {@code tf.constant([0] + sizes)}: the synthesized list has no length, so the constant stays ⊤
   * rather than reading the two-element miscount.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testConcatenatedOffsetsDecline()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_offsets", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A reshape whose target is a concatenated shape vector keeps resolving through the def-use
   * vector walk: the synthesized list is no literal evidence, so the walk runs as before.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReshapeOverConcatenatedShapeVector()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_reshaped",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * {@code np.array([0] * 3)}: no reader derives the one-element miscount.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepeatedArrayDecline()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "decline_np_repeated", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * {@code tf.constant([1, 2] + [3])}: no reader derives the two-element miscount.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testConcatenatedConstantDecline()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "decline_tf_concatenated",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * {@code tf.zeros([2] * 2)}: a shape argument that is a synthesized list is no literal evidence,
   * so the result keeps its unknown shape and float32 dtype rather than reading a one-extent shape,
   * a scalar, or nothing at all.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepeatedShapeArgumentDecline()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "decline_zeros_repeated", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A list beside a tensor is the tensor's own addition: the result reads as the tensor, and the
   * model synthesizes no list for it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMixedAddDecline()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "decline_mixed_add", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A list beside an ndarray is the ndarray's multiplication: the result reads as the array, and
   * the model synthesizes no list for it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMixedMulDecline()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "decline_mixed_mul", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 2))));
  }
}
