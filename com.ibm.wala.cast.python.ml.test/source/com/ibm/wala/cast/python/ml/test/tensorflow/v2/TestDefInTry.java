package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a function defined inside a {@code try} body, whose own body makes a call, does not
 * stop its module from translating (<a href="https://github.com/wala/ML/issues/1004">wala/ML
 * #1004</a>).
 */
public class TestDefInTry extends AbstractTensorTest {

  private static final TensorType T2 = TensorType.of(FLOAT_32, 2);

  /** A function defined in a module-level {@code try}. */
  @Test
  public void testModuleLevel() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_def_in_try.py", "sink", 1, 1, Map.of(2, Set.of(T2)));
  }

  /** A function defined in a method's {@code try} with a {@code finally}. */
  @Test
  public void testInMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_def_in_try_method.py", "sink", 1, 1, Map.of(2, Set.of(T2)));
  }

  /** The defined function's own {@code try} keeps its handler. */
  @Test
  public void testOwnTryKeepsHandler()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_def_in_try_own_try.py", "fallback", 1, 1, Map.of(2, Set.of(T2)));
  }
}
