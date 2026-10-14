package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests reads of class-body attributes through an instance: the instance's attribute is the class
 * body's value, and an inherited one the base class body's, unless an instance write or a method of
 * the same name stands in for it.
 */
public class TestClassAttributeReads extends AbstractTensorTest {

  private static final String FILE = "tf2_test_class_attribute_reads.py";

  /** A method the instance shadows with an attribute: the attribute's tensor, as before. */
  @Test
  public void testShadowedMethod() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "shadow_sink", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A class attribute an instance write overrides: either value, since the analysis does not order
   * the instance write before the read.
   */
  @Test
  public void testOverriddenClassAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "override_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2), TensorType.of(FLOAT_32, 3))));
  }

  /** A class attribute read through the class itself, as before. */
  @Test
  public void testReadThroughClass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "class_read_sink", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /** A class attribute that is an integer, read through the instance as a shape. */
  @Test
  public void testIntegerClassAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "config_sink", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** A base class body's attribute, read through an instance of a subclass. */
  @Test
  public void testInheritedClassAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "inherited_sink", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }
}
