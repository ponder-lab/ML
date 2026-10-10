package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the dtype of a Python float literal times a NumPy array whose dtype the analysis does not
 * resolve. NumPy promotes an integral array to {@code float64} and keeps a {@code float32} one, so
 * the result's dtype is as unknown as the operand's; the fixture's results are {@code float64} at
 * run time, and {@code float32} would be a wrong dtype.
 */
public class TestFloatLiteralUnresolved extends AbstractTensorTest {

  private static final String FILE = "tf2_test_float_literal_unresolved.py";

  /**
   * An array of a value the analysis does not model, {@code np.array(json.loads(...))}, times a
   * float literal.
   *
   * <p>TODO: The analysis reads {@code float32} here, guessing the result's dtype beside an operand
   * whose dtype does not resolve. Flip to a plain {@code @Test} when <a
   * href="https://github.com/wala/ML/issues/1033">wala/ML#1033</a> is fixed.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test(expected = AssertionError.class)
  public void testLoaded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loaded", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * An array of {@code list(map(int, ...))} rows times a float literal, the shape of an array built
   * from parsed annotation fields.
   *
   * <p>TODO: The analysis reads {@code float32} here, guessing the result's dtype beside an operand
   * whose dtype does not resolve. Flip to a plain {@code @Test} when <a
   * href="https://github.com/wala/ML/issues/1033">wala/ML#1033</a> is fixed.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test(expected = AssertionError.class)
  public void testMapped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mapped", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A parameter fed both an array the analysis resolves, {@code np.arange(4)}, and one it does not,
   * times a float literal. The resolved operand promotes to {@code float64}; the unresolved one
   * gives an unknown dtype, never a guessed {@code float32}, which would be a wrong member beside
   * the right one.
   *
   * <p>TODO: The analysis reads a {@code float32} member here, guessing the result's dtype beside
   * the unresolved operand. Flip to a plain {@code @Test} when <a
   * href="https://github.com/wala/ML/issues/1033">wala/ML#1033</a> is fixed.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test(expected = AssertionError.class)
  public void testMixed() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_mixed",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 4), TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }
}
