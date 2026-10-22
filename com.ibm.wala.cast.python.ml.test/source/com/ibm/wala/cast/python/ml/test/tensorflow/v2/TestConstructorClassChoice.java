package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that one call site constructing either of two classes, through a variable bound to one or
 * the other, runs each class's own {@code __init__} and binds each class's own methods on that
 * class's instances only. Before, each class's synthesized constructor read its own initializer and
 * methods off the call site's class-object argument, which holds both classes, so each instance ran
 * both initializers and carried both classes' methods of a name.
 */
public class TestConstructorClassChoice extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_constructor_class_choice.py";

  /** The first class's method reads only the field its own initializer writes. */
  @Test
  public void testFirstClass() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_a", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** The second class's method reads only the field its own initializer writes. */
  @Test
  public void testSecondClass() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_b", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  /**
   * The initializers alone: the two classes' methods have different names, so only the other
   * class's initializer running on this class's instance could give the sink the other tensor.
   */
  @Test
  public void testInitializerFirstClass()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_c", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** As {@link #testInitializerFirstClass()}, for the second class. */
  @Test
  public void testInitializerSecondClass()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_d", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  /**
   * Class-body attributes alone: neither class has an initializer, and their methods have different
   * names, so only the other class's attribute copied onto this class's instance could give the
   * sink the other tensor.
   */
  @Test
  public void testClassAttributeFirstClass()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_e", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** As {@link #testClassAttributeFirstClass()}, for the second class. */
  @Test
  public void testClassAttributeSecondClass()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }
}
