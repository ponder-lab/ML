package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a bare decorator ({@code @d}) is applied as Python applies it, {@code d(f)}, so a
 * decorated function is reached through the wrapper its decorator returns (<a
 * href="https://github.com/wala/ML/issues/188">wala/ML#188</a>). Each decorated function's
 * parameter is checked, which holds only if the function is reached from its decorator's wrapper.
 */
public class TestDecoratorApplication extends AbstractTensorTest {

  private static final String FILE = "tf2_test_decorator_application.py";

  /** The identity decorator returns the function itself. */
  @Test
  public void testIdentity() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "f_ident", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A decorator returning a nested wrapper that calls the captured function. */
  @Test
  public void testWrapper() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "f_plain", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A wrapper decorated with {@code functools.wraps}, reached through the module attribute. */
  @Test
  public void testFunctoolsWraps() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "f_qualified", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A wrapper decorated with {@code wraps}, imported from {@code functools} by name. */
  @Test
  public void testImportedWraps() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "f_bare", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Stacked decorators apply from the {@code def} outward, {@code outer(inner(f))}: the function
   * receives {@code inner}'s int32 cast of {@code outer}'s float32 ones. The reverse order would
   * hand it the float32 ones instead.
   */
  @Test
  public void testStackedOrder() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "f_stacked", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /** A class used as a bare decorator: its instance wraps the function and calls it. */
  @Test
  public void testClassDecorator() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "f_class", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * A multi-module project in which a function is decorated by a {@code functools.wraps}-based
   * decorator defined in a sibling module: the wrapper converts the image to float32 before calling
   * the decorated function.
   */
  @Test
  public void testWrapsAcrossModules()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "wraps_proj/core/__init__.py",
          "wraps_proj/core/convert_type_decorator.py",
          "wraps_proj/core/quality.py",
          "wraps_proj/test_quality.py"
        },
        "core/quality.py",
        "consume",
        "wraps_proj",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 4, 3))));
  }
}
