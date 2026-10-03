package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
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

  /**
   * A class with two program-defined bases has its first declared base as its superclass
   * (wala/ML#1014). The superclass was the first base in the iteration order of a hash set of the
   * bases, which varies from run to run, so the `super()` body built from it bound one base's
   * methods in some runs and the other's in others. Twelve such classes make a wrong pick for one
   * of them all but certain under that order.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testFirstDeclaredBaseIsTheSuperclass() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_superclass_order.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    IClassHierarchy cha = builder.getClassHierarchy();
    for (int k = 0; k < 12; k++) {
      IClass both =
          cha.lookupClass(
              TypeReference.findOrCreate(
                  PythonTypes.pythonLoader, "Lscript tf2_test_superclass_order.py/Both" + k));
      assertNotNull("Both" + k + " is defined", both);
      assertEquals(
          "Both" + k + "'s superclass",
          "Lscript tf2_test_superclass_order.py/First" + k,
          both.getSuperclass().getName().toString());
    }
  }

  /**
   * `super().m()` in each two-base class reaches the first base's `m`, as Python's method
   * resolution order requires (wala/ML#1014), so only the first bases' {@code (2, 2)} arrives.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSuperReachesTheFirstBase()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_superclass_order.py",
        "consume_first",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }
}
