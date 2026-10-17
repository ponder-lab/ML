package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that applying a program class that subclasses a built-in Keras layer has the type the
 * built-in layer gives its input (wala/ML#1108). The application runs the built-in layer's {@code
 * call} when the class inherits it, and the override's {@code super().call} reaches it when the
 * class overrides {@code call}. Before, the inherited form had no call-graph target, and the
 * built-in layer's {@code call} summary had no generator, so neither application was a tensor.
 */
public class TestBuiltinLayerSubclassApply extends AbstractTensorTest {

  private static final String FILE = "tf2_test_builtin_layer_subclass_apply.py";

  /**
   * A subclass of {@code BatchNormalization} that overrides {@code call} and returns {@code
   * super().call(x, training)}: the application is the input's type.
   */
  @Test
  public void testOverrideCallingSuper()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_frozen", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A subclass of {@code BatchNormalization} that inherits {@code call}: the input's type. */
  @Test
  public void testInheritedCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_plain", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * A subclass of {@code Dense} that inherits {@code call} and its constructor: {@code Dense}'s
   * transform of the input, with the size the subclass is constructed with. The subclass's
   * synthesized constructor does not write {@code Dense}'s {@code units} field, so the size is read
   * from the constructor's call; before, the unread size gave the input's own shape.
   */
  @Test
  public void testInheritedDenseCall()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_wider", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The same subclass constructed with its size by keyword, after another keyword: the size is the
   * argument the {@code units} keyword names, not the first argument the call passes.
   */
  @Test
  public void testInheritedDenseCallKeywordSize()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5))));
  }

  /**
   * The same subclass constructed through a program factory, {@code make(cls, n)}, with two
   * different sizes: the factory's {@code cls(n)} is the constructor's one call, and its size is
   * the factory's parameter, so each application gets the join of both sizes rather than one size
   * alone.
   */
  @Test
  public void testFactoryJoinsSizes() throws ClassHierarchyException, CancelException, IOException {
    Set<TensorType> joined = Set.of(TensorType.of(FLOAT_32, 2, 6), TensorType.of(FLOAT_32, 2, 7));
    test(FILE, "consume_made_six", 1, 1, Map.of(2, joined));
    test(FILE, "consume_made_seven", 1, 1, Map.of(2, joined));
  }

  /**
   * A subclass of {@code Dense} that defines {@code __init__} and passes the size on through {@code
   * super().__init__(n)}: {@code Dense}'s transform of the input. The size is not read through the
   * {@code super().__init__} call, and an unread size gives the input's own shape.
   *
   * <p>TODO: Flip to a plain {@code @Test} once https://github.com/wala/ML/issues/1112 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testInitPassingSize() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sized", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
