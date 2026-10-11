package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that constructing a class without its own {@code __init__} runs the nearest one in its
 * method resolution order (<a href="https://github.com/wala/ML/issues/1048">wala/ML#1048</a>).
 * Before, the synthesized constructor looked only for the class's own initializer, so it called
 * nothing and dropped its arguments, and a field the inherited initializer sets read as empty.
 */
public class TestInheritedInit extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_inherited_init.py";

  /** A subclass of a plain class. */
  @Test
  public void testPlainBase() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 4))));
  }

  /** A subclass of a class deriving from {@code object}, with a method of its own. */
  @Test
  public void testObjectBase() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_object_base", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /** The first base defines no initializer, so the second base's runs. */
  @Test
  public void testSecondBase() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_second_base", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Neither the class nor its parent defines one, so the grandparent's runs. */
  @Test
  public void testGrandparent() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_grandparent", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 7))));
  }

  /** A subclass of a program-defined Keras model, the constructor form of wala/ML#1000. */
  @Test
  public void testKerasBase() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_keras", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6))));
  }
}
