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
