package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.NoneType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a parameter that {@code None} reaches, and that receives tensors in another context,
 * has {@code None} among its types (<a
 * href="https://github.com/wala/ML/issues/1040">wala/ML#1040</a>).
 */
public class TestNoneParameter extends AbstractTensorTest {

  /** A {@code None} default left by one call, beside a call that passes a tensor. */
  @Test
  public void testOmittedNoneDefault()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "optional",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32, NoneType.INSTANCE)));
  }

  /** A {@code None} passed explicitly, beside a call that passes a tensor. */
  @Test
  public void testExplicitNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "explicit",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3), NoneType.INSTANCE)));
  }

  /** A {@code None} default forwarded through another function's parameter by keyword. */
  @Test
  public void testForwardedNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "build_mask",
        2,
        6,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 4, 5)),
            3,
            Set.of(TensorType.of(INT_32, 4), NoneType.INSTANCE)));
  }

  /** A parameter that only ever receives {@code None} keeps no state. */
  @Test
  public void testNeverTensor() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "never_tensor", 0, 0, Map.of());
  }

  /** A parameter that {@code None} never reaches has no {@code None} member. */
  @Test
  public void testAlwaysTensor() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "always_tensor", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * A loop variable read after its loop is unbound before the loop's first iteration, a path the IR
   * gives the null constant; no call passes {@code None} there.
   */
  @Test
  public void testUnboundAfterLoop() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "after_loop", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A variable bound on one branch only is unbound, not {@code None}, on the other. */
  @Test
  public void testConditionallyBound()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "maybe_bound", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * A call's result that a program function returns as {@code None}. The same function's tensor
   * return reaches the parameter with its dtype unknown, a separate imprecision.
   */
  @Test
  public void testReturnedNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "returned",
        1,
        2,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TensorType.of(UNKNOWN, 2), NoneType.INSTANCE)));
  }

  /** An attribute the program sets to {@code None}, read and passed. */
  @Test
  public void testFieldNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "from_field",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32, NoneType.INSTANCE)));
  }

  /** A dictionary entry the program sets to {@code None}, read and passed. */
  @Test
  public void testEntryNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "from_entry",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32, NoneType.INSTANCE)));
  }

  /**
   * A variable that is {@code None} on one branch and a tensor on the other: the IR merges the two
   * in a phi whose {@code None} operand is the null constant it also gives an unbound path, so the
   * {@code None} is not decided and the parameter has no {@code None} member.
   *
   * <p>TODO: Known gap, tracked by https://github.com/wala/ML/issues/1040.
   */
  @Test
  public void testPhiNoneUndecided() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "from_phi",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * A {@code None} inside a starred argument: a call with a starred argument at or before the
   * parameter decides nothing, so the parameter has no {@code None} member.
   *
   * <p>TODO: Known gap, tracked by https://github.com/wala/ML/issues/1040.
   */
  @Test
  public void testStarredNoneUndecided()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "from_star",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A parameter forwarded only where {@code mask is not None} holds receives no {@code None}. */
  @Test
  public void testIsNotNoneGuard() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "guarded_inner", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** The {@code None} default the guarded caller is left to is still among its own types. */
  @Test
  public void testGuardedCallerKeepsNone()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "guarded_outer",
        2,
        5,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32, NoneType.INSTANCE)));
  }

  /**
   * A parameter forwarded on the {@code else} arm of {@code mask is None} receives no {@code None}.
   */
  @Test
  public void testIsNoneElseGuard() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "else_inner", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A parameter forwarded only where it tests true receives no {@code None}. */
  @Test
  public void testTruthGuard() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "truth_inner",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /**
   * A field set to {@code None} in {@code __init__} and to a tensor in {@code build}, the Keras
   * idiom: the {@code call} that reads it runs after {@code build}, so no call passes {@code None}
   * on. A flow-insensitive read cannot order the two stores, so only a read every value of which is
   * {@code None} counts.
   */
  @Test
  public void testFieldReassignedInBuild()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_none_parameter.py", "kernel_sink", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A parameter left to a default that is not {@code None} receives no {@code None}. */
  @Test
  public void testNonNoneDefault() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "scaled",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A method's {@code None} default, left by one call through the method's trampoline. */
  @Test
  public void testMethodNoneDefault() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "Masker.apply",
        2,
        3,
        Map.of(3, Set.of(TENSOR_2_FLOAT32), 4, Set.of(TENSOR_2_FLOAT32, NoneType.INSTANCE)));
  }

  /** A method's default that is not {@code None}, left by one call, receives no {@code None}. */
  @Test
  public void testMethodNonNoneDefault()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_none_parameter.py",
        "Masker.scale",
        2,
        3,
        Map.of(3, Set.of(TENSOR_2_FLOAT32), 4, Set.of(TENSOR_2_FLOAT32)));
  }
}
