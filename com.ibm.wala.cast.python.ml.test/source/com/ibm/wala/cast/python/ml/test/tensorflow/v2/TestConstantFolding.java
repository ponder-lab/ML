package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a tensor built from a binary operation over two literals takes the operation's own
 * type (<a href="https://github.com/wala/ML/issues/1143">wala/ML#1143</a>).
 */
public class TestConstantFolding extends AbstractTensorTest {

  private static final String FILE = "tf2_test_constant_folding.py";

  /** The sum {@code 0.5 + 0.25} is a float, so its tensor is float32. */
  @Test
  public void testSum() throws Exception {
    test(FILE, "consume_sum", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** The true quotient {@code 1 / 2} of two integers is a float, so its tensor is float32. */
  @Test
  public void testQuotient() throws Exception {
    test(FILE, "consume_quotient", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * The comparison {@code 1 < 2} is a bool, so its tensor is bool. It isn't folded, so its tensor
   * reads as unknown.
   *
   * <p>TODO: The comparison's result has no instance key, so {@code tf.constant} reads no value
   * from it. Flip to a plain {@code @Test} once a binary operation's result has one (<a
   * href="https://github.com/wala/ML/issues/805">wala/ML#805</a>).
   */
  @Test(expected = AssertionError.class)
  public void testComparison() throws Exception {
    test(FILE, "consume_comparison", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_BOOL)));
  }

  /** The sum {@code 1 + 1} still folds to an integer, so its tensor is int32. */
  @Test
  public void testIntSum() throws Exception {
    test(FILE, "consume_int_sum", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** A float literal, with nothing to fold, gives float32. */
  @Test
  public void testLiteral() throws Exception {
    test(FILE, "consume_literal", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }
}
