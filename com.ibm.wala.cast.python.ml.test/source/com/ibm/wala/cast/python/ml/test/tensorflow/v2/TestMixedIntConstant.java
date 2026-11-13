package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
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
}
