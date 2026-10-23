package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Partial dtype unions (wala/ML#862). A value joined in one frame from an allocation whose dtype is
 * known and one whose dtype the analysis cannot read has the known dtype beside {@code unknown}.
 * The value read over the value's points-to set collapsed any set with an unknown member to {@code
 * unknown} alone, so an operation reading its operand that way lost the known member, and the
 * result read as unknown although every allocation it could come from that did resolve was known.
 * The partial union now survives the read, and each operation keeps the known member beside the
 * unknown one; a fully known and a fully unknown operand read as before.
 */
public class TestDTypePartialUnion extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_dtype_partial.py";

  private static final TensorType SCALAR_TENSOR_OF_UNKNOWN = TensorType.of(UNKNOWN);

  /**
   * The joined operand itself, which the dataflow state already carries as both members: the
   * control that the fixture forms the partial union.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPartialOperand() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_partial_operand",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(UNKNOWN, 2, 3))));
  }

  /**
   * A transposition of an identity of the partial operand keeps both members. Master reads {@code
   * unknown} alone: the transposition reads its operand, the identity's result, over the identity's
   * points-to set, whose two allocations' dtypes collapsed.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPartialPassThrough()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_partial_pass_through",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2), TensorType.of(UNKNOWN, 3, 2))));
  }

  /**
   * A maximum over a partial int32 operand plus one keeps int32 beside unknown. Master reads {@code
   * unknown} alone, the reduction's operand read collapsing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPartialReduce() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_partial_reduce",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32, SCALAR_TENSOR_OF_UNKNOWN)));
  }

  /**
   * A product with a float literal keeps both members: a control, read the same on master, whose
   * result's dtypes also arrive through the operand's dataflow state.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPartialElementwise()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_partial_elementwise",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(UNKNOWN, 2, 3))));
  }

  /**
   * A cast names its result dtype, so the operand's partial dtype does not reach it: a control.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPartialCast() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_partial_cast", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }

  /**
   * A product with a float32 kernel is float32 alone, since the runtime requires the operands to
   * agree on dtype and the kernel resolves: a partial operand does not shadow a resolved one. The
   * unknown shape is master's and not this change's.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPartialMatMul() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE, "consume_partial_matmul", 1, 1, Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }

  /**
   * Both allocations known: the transposition reads float32 alone, unchanged.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testKnownPassThrough() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_known_pass_through",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2))));
  }

  /**
   * Both allocations known: the reduction reads int32 alone, unchanged.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testKnownReduce() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_known_reduce", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Neither allocation known: the transposition reads unknown alone, unchanged.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnknownPassThrough()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_unknown_pass_through",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UNKNOWN, 3, 2))));
  }

  /**
   * Neither allocation known: the reduction reads unknown alone, unchanged.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnknownReduce() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_unknown_reduce", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_UNKNOWN)));
  }
}
