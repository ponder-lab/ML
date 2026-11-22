package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Guards whose operand may take a value no points-to set holds, so neither arm is decided: a field
 * rewritten from a parsed configuration, and a call whose callee may fall off its end.
 */
public class TestUnseenGuardValues extends AbstractTensorTest {

  @Test
  public void testFieldReadFromParsedConfig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_flagged",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testControlUnguardedMethodCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_unguarded",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testControlBeforeGuard()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_implicit_none_guard.py",
        "consume_before_guard",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testFrameFieldReadFromParsedConfig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_checked",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testControlGuardOnExplicitReturnDecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_implicit_none_guard.py", "consume_never", 0, 0, Map.of());
  }

  @Test
  public void testControlGuardOnReturnInEndlessLoopDecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_implicit_none_guard.py", "consume_never_looped", 0, 0, Map.of());
  }

  @Test
  public void testMethodFieldReadFromParsedConfig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_method_checked",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testGuardOnRecursiveCalleesEnds()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_implicit_none_guard.py",
        "consume_after_recursive_guard",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testGlobalWrittenFromParsedConfig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_mode",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testParameterFromMergeOfParsedConfig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_phi_checked",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A callee whose only fall-off block a constant guard of its own never reaches returns True
   * alone, so the guard on its result is still decided.
   */
  @Test
  public void testControlGuardOnFallOffAConstantGuardKills()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_implicit_none_guard.py", "consume_never_in_mode", 0, 0, Map.of());
  }

  /**
   * A callee falls off its end after a statement, not a branch, so the guard on its result
   * declines.
   */
  @Test
  public void testAfterGuardOnFallOffEndingInAStatement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_implicit_none_guard.py",
        "consume_after_logged_guard",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A callee ends in a bare return, which returns None, so the guard on its result declines. */
  @Test
  public void testAfterGuardOnBareReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_implicit_none_guard.py",
        "consume_after_bare_guard",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A flag read from a parsed configuration reaches the guarded parameter through a starred call,
   * whose arguments the walk cannot place, so the guard declines.
   */
  @Test
  public void testStarredArgumentFromParsedConfig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_starred",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A phi of two field reads whose every write is seen passes the guarded parameter a seen value,
   * so the guard is still decided.
   */
  @Test
  public void testControlMergeOfSeenFieldsDecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_field_config_flag.py", "consume_never_seen", 0, 0, Map.of());
  }

  /**
   * A parameter a function forwards to itself, and whose one outside caller passes a constant, is
   * seen, so the guard is still decided.
   */
  @Test
  public void testControlRecursiveForwardDecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_field_config_flag.py", "consume_never_counted", 0, 0, Map.of());
  }

  /**
   * A parameter forwarded through seventy functions takes the walk past its bound of 64 values,
   * which declines. The consumer reads its argument's type although no run calls it, a precision
   * the bound gives up.
   */
  @Test
  public void testWalkPastItsBoundDeclines()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_field_config_flag.py",
        "consume_far",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A callee whose only fall-off block, after a statement, a constant guard of its own never
   * reaches returns True alone, so the guard on its result is still decided.
   */
  @Test
  public void testControlGuardOnDeadFallOffStatementDecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_implicit_none_guard.py", "consume_never_in_logged_mode", 0, 0, Map.of());
  }

  @Test
  public void testAfterGuardOnImplicitNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_implicit_none_guard.py",
        "consume_after_guard",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testLoopAfterGuardOnImplicitNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_implicit_none_guard.py",
        "consume_in_loop_after_guard",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
