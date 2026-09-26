package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A guard over a {@code @click.option} default decides nothing, while a shape read of the same
 * default keeps its value (wala/ML#971). The default is the value of the one invocation that passes
 * no option; every other invocation the command line admits binds the parameter otherwise, so a
 * comparison fold that decided the guard from it would prune arms the program runs. A Python
 * default is different: it is the only binding of a parameter no call passes, and the fold keeps
 * deciding those.
 */
public class TestClickGuard extends AbstractTensorTest {

  private static final String FILE = "tf2_test_click_guard.py";

  /**
   * A φ whose arms a click-defaulted flag selects keeps both arms: the flag's default {@code False}
   * decides nothing.
   */
  @Test
  public void testClickGuardedMergeKeepsBothArms()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_merge",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 4))));
  }

  /** A call site a click-defaulted flag guards delivers its argument, with its dtype. */
  @Test
  public void testClickGuardedCallDelivers()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_call", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /** The same allocation unguarded: the control for the guarded call's dtype. */
  @Test
  public void testUnguardedControl() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_plain", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * A click-defaulted flag read back from an attribute decides nothing: the field rule declines.
   */
  @Test
  public void testClickDefaultOnAttributeKeepsBothArms()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_field",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 4))));
  }

  /** A click-defaulted flag passed as an argument decides nothing at that call site. */
  @Test
  public void testClickDefaultAsArgumentKeepsBothArms()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_arg",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 4))));
  }

  /** A string click default decides nothing, and a method call on it still dispatches. */
  @Test
  public void testStringClickDefaultKeepsBothArms()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_mode",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2), TensorType.of(INT_32, 3))));
  }

  /** A tuple click default is no constant key; it passes through and reads as a shape. */
  @Test
  public void testTupleClickDefaultReadsAsShape()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A shape read of a click default keeps the default's value (wala/ML#875). */
  @Test
  public void testClickDefaultShapeKept()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6))));
  }

  /** A click default flowing through arithmetic into a shape keeps folding as a size. */
  @Test
  public void testClickDefaultArithmeticShapeKept()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arith", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 12))));
  }
}
