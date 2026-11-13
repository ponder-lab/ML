package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A Python {@code int} whose points-to set holds a constant and the key of an {@code int} of
 * unknown value, such as a loop counter, has no single value, so a size computed from it has an
 * unknown extent rather than the one the constant alone gives (<a
 * href="https://github.com/wala/ML/issues/1252">wala/ML#1252</a>).
 */
public class TestMixedIntConstant extends AbstractTensorTest {

  private static final String FILE = "tf2_test_mixed_int_constant.py";

  /** An array of {@code float64} whose shape is not known. */
  private static final TensorType TENSOR_UNKNOWN_SHAPE_FLOAT64 = new TensorType(FLOAT_64, null);

  /** A vector of {@code float64} whose one extent is not known. */
  private static final TensorType TENSOR_UNRESOLVED_FLOAT64 =
      new TensorType(FLOAT_64, List.of(UnresolvedDim.INSTANCE));

  /** A vector whose one extent is not known. */
  private static final TensorType TENSOR_UNRESOLVED_FLOAT32 =
      new TensorType(FLOAT_32, List.of(UnresolvedDim.INSTANCE));

  /** The constant {@code 0} on one branch and an {@code int(...)} on the other. */
  @Test
  public void testMixed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /** The constant {@code 0} on one branch and a {@code len(...)} on the other. */
  @Test
  public void testMixedLen() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_len", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /** A loop counter from {@code 0}. */
  @Test
  public void testCounter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_counter", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /**
   * The constant {@code 3} on one branch and a tensor's static shape on the other, an allocator's
   * shape that is no single constant, so its shape is unknown.
   */
  @Test
  public void testConstantOrShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_constant_or_shape", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT64)));
  }

  /**
   * An allocator's shape that is the constant {@code 0} or an {@code int(...)}, an {@code int} of
   * unknown value, which allocates rank 1 whatever its value (wala/ML#1159).
   */
  @Test
  public void testZerosOfMixed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zeros_of_mixed", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT64)));
  }

  /**
   * A guard on a parameter a call passes the constant {@code 0} or an {@code int(...)}, which does
   * not fold, so both arms reach the consumer.
   */
  @Test
  public void testGuarded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_guarded", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_3_FLOAT32)));
  }

  /**
   * A {@code squeeze} whose {@code axis} is {@code 1} or an {@code int} of unknown value, supplied
   * but not read as a constant, so it may name any axes and the shape is unknown rather than every
   * size-1 axis squeezed, the omitted axis's reading (wala/ML#1022).
   */
  @Test
  public void testSqueezeAxis() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_squeeze", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A {@code one_hot} whose {@code depth} is {@code 4} or an {@code int} of unknown value: an
   * unresolved depth, rather than no tensor (wala/ML#1022).
   */
  @Test
  public void testOneHotDepth() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_one_hot_depth",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, List.of(new NumericDim(2), UnresolvedDim.INSTANCE)))));
  }

  /**
   * A {@code one_hot} whose {@code axis} is {@code 0} or an {@code int} of unknown value, which may
   * put the depth at any position, rather than the last, the omitted axis's (wala/ML#1022).
   */
  @Test
  public void testOneHotAxis() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_one_hot_axis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 2), TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A {@code boolean_mask} whose {@code axis} is {@code 1} or an {@code int} of unknown value,
   * which may mask any axis the ranks admit, rather than the first, the omitted axis's
   * (wala/ML#1022).
   */
  @Test
  public void testBooleanMaskAxis() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_boolean_mask",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32, List.of(DynamicDim.INSTANCE, new NumericDim(3), new NumericDim(4))),
                new TensorType(
                    FLOAT_32, List.of(new NumericDim(2), DynamicDim.INSTANCE, new NumericDim(4))),
                new TensorType(
                    FLOAT_32,
                    List.of(new NumericDim(2), new NumericDim(3), DynamicDim.INSTANCE)))));
  }
}
