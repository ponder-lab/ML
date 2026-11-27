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

  private static final String CREATED_IN_SCOPE = "tf2_test_floatx_created_in_scope.py";

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
   * A layer created between a {@code set_floatx("float64")} and its restore, called after the
   * restore: Keras read {@code floatx()} when it created the layer, so the output is {@code
   * float64}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCreatedInScopeOutput()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        CREATED_IN_SCOPE,
        "consume_output",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(FLOAT_64, 2, 3))));
  }

  /**
   * A weight a user layer adds without a {@code dtype} in {@code build}, where the layer was
   * created between a {@code set_floatx("float64")} and its restore and is built after the restore:
   * the weight takes the dtype the layer read when it was created.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCreatedInScopeWeight()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        CREATED_IN_SCOPE,
        "consume_weight",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(FLOAT_64, 2, 3))));
  }

  /**
   * A user layer's {@code call} input, which Keras casts to the compute dtype the layer read when
   * it was created, between a {@code set_floatx("float64")} and its restore, though it is called
   * after the restore.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCreatedInScopeCallInput()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        CREATED_IN_SCOPE,
        "consume_call_input",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2), TensorType.of(FLOAT_64, 2, 2))));
  }

  /**
   * A {@code floatx()} token taken between a {@code set_floatx("float64")} and its restore, passed
   * as a {@code dtype} after the restore: the token holds the value {@code floatx()} returned.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCreatedInScopeToken()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        CREATED_IN_SCOPE,
        "consume_token_use",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2), TensorType.of(FLOAT_64, 2, 2))));
  }

  /**
   * A {@code Dense} output outside a function whose restore an exception can skip, to a handler in
   * the same function, which leaves its setting in effect everywhere.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSkippedRestore() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_floatx_skipped_restore.py",
        "consume_outside",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(FLOAT_64, 2, 4))));
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
