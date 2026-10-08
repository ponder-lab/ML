package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that calling a Keras layer whose {@code call} is inherited from a user base class reaches
 * that {@code call}, as {@code Layer.__call__} finds {@code call} along the method resolution
 * order. The dispatch looked {@code call} up on the instance's own class only, so a subclass that
 * inherits it had no target.
 */
public class TestInheritedKerasCall extends AbstractTensorTest {

  private static final String FILE = "tf2_test_inherited_keras_call.py";

  private static final TensorType INPUT = TensorType.of(FLOAT_32, 2, 3, 4);

  /** A subclass with no body of its own, called at a site only it reaches. */
  @Test
  public void testInheritedCallSingleSite()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_single", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /** A subclass with its own {@code __init__} delegating to its base's. */
  @Test
  public void testInheritedCallOwnInit()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_own_init", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /** The {@code call} two program-defined bases up. */
  @Test
  public void testInheritedCallGrandparent()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_grandparent", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /**
   * The subclass in a list of layers a loop calls in turn, so the site sees several classes; the
   * dense layer before it gives its input four units.
   */
  @Test
  public void testInheritedCallMixedSite()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /** Control: a subclass's own {@code call} still overrides its base's. */
  @Test
  public void testOverridingCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_override", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /**
   * Control: a subclass's own {@code call} still overrides its base's at the site several classes
   * reach, beside a sibling that inherits the base's.
   */
  @Test
  public void testOverridingCallMixedSite()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_override", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /**
   * The {@code call} of a second declared base, after a first base, a plain mixin, that declares
   * none: Python's method resolution order reaches the second base, where the first base's own
   * chain ends.
   */
  @Test
  public void testInheritedCallAfterMixin()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixin", 1, 1, Map.of(2, Set.of(INPUT)));
  }
}
