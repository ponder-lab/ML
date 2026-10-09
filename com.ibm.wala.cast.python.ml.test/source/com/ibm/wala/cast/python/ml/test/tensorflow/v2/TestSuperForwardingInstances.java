package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a base constructor reached through {@code super(...).__init__(...)}, explicit or
 * zero-argument, from a subclass constructor writes to the one instance under construction (<a
 * href="https://github.com/wala/ML/issues/1023">wala/ML#1023</a>). Two wrappers built around
 * different layers in one method each forward their call to their own layer, so each call's result
 * has its own layer's width.
 */
public class TestSuperForwardingInstances extends AbstractTensorTest {

  /** The first wrapper's call result, through its own eight-unit layer. */
  @Test
  public void testFirstWrapper() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_forwarding_instances.py",
        "consume_a",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 8))));
  }

  /** The second wrapper's call result, through its own three-unit layer. */
  @Test
  public void testSecondWrapper() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_forwarding_instances.py",
        "consume_b",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The first wrapper's call result when the constructors forward through a zero-argument {@code
   * super()}.
   */
  @Test
  public void testFirstWrapperImplicitSuper()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_forwarding_instances_implicit.py",
        "consume_a",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 8))));
  }

  /**
   * The second wrapper's call result when the constructors forward through a zero-argument {@code
   * super()}.
   */
  @Test
  public void testSecondWrapperImplicitSuper()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_forwarding_instances_implicit.py",
        "consume_b",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
