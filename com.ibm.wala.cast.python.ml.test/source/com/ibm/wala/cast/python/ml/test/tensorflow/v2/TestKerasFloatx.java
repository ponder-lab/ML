package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Keras dtypes that default to {@code tf.keras.backend.floatx()} (<a
 * href="https://github.com/wala/ML/issues/871">wala/ML#871</a>). A reachable {@code
 * tf.keras.backend.set_floatx} call adds what it may set to the {@code float32} default: everywhere
 * when no function restores the default after it, and only in what runs between the two calls when
 * its function restores the default on every normal path to its exit.
 */
public class TestKerasFloatx extends AbstractTensorTest {

  private static final String IMPLICIT = "tf2_test_floatx_implicit.py";

  private static final String SCOPED = "tf2_test_floatx_scoped.py";

  /**
   * An {@code Input} without a {@code dtype}, after an unrestored {@code set_floatx("float64")}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testInputDefault() throws ClassHierarchyException, CancelException, IOException {
    test(
        IMPLICIT,
        "consume_input",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(7))),
                new TensorType(FLOAT_64, asList(DynamicDim.INSTANCE, new NumericDim(7))))));
  }

  /**
   * A {@code Dense} layer's output, its compute dtype, after an unrestored {@code
   * set_floatx("float64")}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDenseOutput() throws ClassHierarchyException, CancelException, IOException {
    test(
        IMPLICIT,
        "consume_dense",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5), TensorType.of(FLOAT_64, 2, 5))));
  }

  /**
   * A user layer's {@code call} input, which Keras casts to the layer's compute dtype, after an
   * unrestored {@code set_floatx("float64")}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCallInputCast() throws ClassHierarchyException, CancelException, IOException {
    test(IMPLICIT, "consume_call", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_FLOAT64)));
  }

  /**
   * A {@code Dense} output between a {@code set_floatx("float64")} and its restore.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testScopedInside() throws ClassHierarchyException, CancelException, IOException {
    test(
        SCOPED,
        "consume_float64",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_64, 2, 4))));
  }

  /**
   * A {@code Dense} output between a {@code set_floatx} of a value the analysis cannot read, a
   * dtype's name, and its restore.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testScopedUnreadable() throws ClassHierarchyException, CancelException, IOException {
    test(
        SCOPED,
        "consume_unreadable",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(UNKNOWN, 2, 4))));
  }

  /**
   * A {@code Dense} output in a function that runs outside both scoped settings keeps the default.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testScopedOutside() throws ClassHierarchyException, CancelException, IOException {
    test(SCOPED, "consume_outside", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A {@code Dense} output outside a function that restores the default on one path only, which
   * leaves its setting in effect everywhere.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnrestoredOnOnePath()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_floatx_unrestored.py",
        "consume_outside",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_64, 2, 4))));
  }
}
