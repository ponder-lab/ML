package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the type of {@code tf.numpy_function}'s and {@code tf.py_function}'s result: the dtype its
 * {@code Tout} argument states, with an unknown shape, since the call does not state the shape the
 * function returns.
 */
public class TestNumpyPyFunction extends AbstractTensorTest {

  private static final String FILE = "tf2_test_numpy_py_function.py";

  /** A {@code float32} tensor of unknown shape. */
  private static final TensorType FLOAT_32_UNKNOWN_SHAPE = new TensorType(FLOAT_32, null);

  /** A single {@code Tout}. */
  @Test
  public void testNumpyFunction() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "single_sink", 1, 1, Map.of(2, Set.of(FLOAT_32_UNKNOWN_SHAPE)));
  }

  /**
   * A list {@code Tout}, whose results are unpacked: each result is one of the list's dtypes, not
   * the first's alone.
   */
  @Test
  public void testNumpyFunctionListFirst()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "first_sink",
        1,
        1,
        Map.of(2, Set.of(FLOAT_32_UNKNOWN_SHAPE, new TensorType(INT_32, null))));
  }

  /** The second result of the same call. */
  @Test
  public void testNumpyFunctionListSecond()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "second_sink",
        1,
        1,
        Map.of(2, Set.of(FLOAT_32_UNKNOWN_SHAPE, new TensorType(INT_32, null))));
  }

  /** A {@code tf.py_function} with a single {@code Tout}. */
  @Test
  public void testPyFunction() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "py_sink", 1, 1, Map.of(2, Set.of(FLOAT_32_UNKNOWN_SHAPE)));
  }
}
