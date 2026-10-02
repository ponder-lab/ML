package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a class call's keywords naming no formal of {@code __init__} reach its {@code
 * **kwargs} (<a href="https://github.com/wala/ML/issues/997">wala/ML#997</a>) and that its
 * positional arguments past {@code __init__}'s formals reach its {@code *args} (<a
 * href="https://github.com/wala/ML/issues/188">wala/ML#188</a>), as a plain function's do. The
 * synthesized constructor declared no star formals, so the class call's packers never ran, and the
 * constructor forwarded nothing for either.
 */
public class TestConstructorStarFormals extends AbstractTensorTest {

  /** A keyword naming no formal, read off {@code **kwargs} inside {@code __init__}. */
  @Test
  public void testKeywordReachesKwargs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ctor_star_formals.py",
        "consume_kwargs_read",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  /** The same pack forwarded by {@code **} into a callee's named formal. */
  @Test
  public void testKwargsForwardsToNamedFormal()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ctor_star_formals.py",
        "consume_forwarded",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  /** A second positional argument to a class whose {@code __init__} takes only {@code *args}. */
  @Test
  public void testPositionalExtraReachesArgs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ctor_star_formals.py",
        "consume_second_positional",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * The motivating shape: a model class taking the kernel only through {@code **kwargs} and
   * forwarding it with {@code super().__init__(**kwargs)} to a base class that names it; the base's
   * write lands and the model trains through the kernel, whose constraint sees the weight.
   */
  @Test
  public void testKwargsThroughSuperTrains()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_kwargs_through_super.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }
}
