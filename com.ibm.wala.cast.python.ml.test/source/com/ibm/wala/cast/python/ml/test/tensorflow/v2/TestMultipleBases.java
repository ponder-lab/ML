package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a class with several program-defined bases reaches the methods of every base when they
 * are called on its instance (<a href="https://github.com/wala/ML/issues/1006">wala/ML#1006</a>):
 * the synthesized constructor bound instance methods along the class's single recorded superclass
 * chain only, so the other bases' methods resolved only through the class object's copied fields,
 * and which of them dispatched varied with the engine version and the call site.
 */
public class TestMultipleBases extends AbstractTensorTest {
  @Test
  public void testSecondBase() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_multiple_bases.py",
        "consume_second_base",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }

  @Test
  public void testThirdBase() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_multiple_bases.py",
        "consume_third_base",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  @Test
  public void testThirdViaHelper() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_multiple_bases.py",
        "consume_third_via_helper",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 4))));
  }

  @Test
  public void testPlainThird() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_multiple_bases.py",
        "consume_plain_third",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 5, 5))));
  }

  /**
   * A diamond: {@code D(C, B)} with {@code B(A)} and {@code C(A)}, where {@code B} overrides {@code
   * A}'s method. Python's method resolution order is D, C, B, A, so the instance runs {@code B}'s
   * method; a depth-first first-occurrence walk would bind {@code A}'s.
   */
  @Test
  public void testDiamondBindsNearestOverride()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_multiple_bases.py",
        "consume_diamond",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 6, 6))));
  }

  /**
   * The shadowed base method is reached only by its direct non-tensor call, not through the
   * diamond.
   */
  @Test
  public void testDiamondShadowsBaseMethod()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_multiple_bases.py", "consume_shadowed", 0, 0);
  }
}
