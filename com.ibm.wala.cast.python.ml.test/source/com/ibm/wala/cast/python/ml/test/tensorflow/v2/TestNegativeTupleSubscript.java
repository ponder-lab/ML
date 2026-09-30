package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A negative constant subscript of a tuple (wala/ML#988), on {@code
 * tf2_test_negative_tuple_subscript.py}. A tuple's elements are fields named by their index from
 * {@code 0}, so {@code shape[-1]} read a field no tuple has and its value was empty: a shape
 * argument written that way ({@code size=shape[-1]}, {@code np.zeros(shape[-2])}) left its result
 * without a rank. The pointer analysis now reads {@code t[-k]} as element {@code n - k} of a tuple
 * of known length {@code n}.
 */
public class TestNegativeTupleSubscript extends AbstractTensorTest {

  private static final String FILE = "tf2_test_negative_tuple_subscript.py";

  /** A {@code randint} draw sized by {@code shape[-1]}, passed to a Keras layer's {@code call}. */
  @Test
  public void testLayerParameter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_identifiers", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /** The module-level {@code np.random.randint} draw sized by {@code shape[-1]}. */
  @Test
  public void testDraw() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_draw", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /** {@code np.zeros(shape[-2])}: a second negative index, the first element. */
  @Test
  public void testZeros() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zeros", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 2))));
  }

  /** Control: a positive subscript, read as before. */
  @Test
  public void testPositiveSubscript() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_positive", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 4))));
  }

  /**
   * Control: a list's negative subscript is not read, since a list's length can change after it is
   * built, so the result's shape stays unknown.
   */
  @Test
  public void testListSubscriptUnread()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_list", 1, 1, Map.of(2, Set.of(new TensorType(FLOAT_64, null))));
  }

  /**
   * Control: a tuple built by concatenation is allocated by the operation (wala/ML#960), not by a
   * literal, so its length is unknown at the allocation. Its negative subscript is not read as one
   * element, and the operation model still reads it as all of the tuple's elements, as before.
   * Reading this allocation's length used to throw, since {@code IR.getNew} throws for a site that
   * is not a {@code new} of that IR.
   */
  @Test
  public void testConcatenatedTupleUnread()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_concatenated",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 4), TensorType.of(FLOAT_64, 2))));
  }

  /** A tuple literal subscripted directly, read without a side effect on its points-to set. */
  @Test
  public void testLiteralSubscript() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_literal", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 4))));
  }

  /**
   * Control: a negative subscript past the tuple's start denotes no element, so nothing is read and
   * the shape stays unknown.
   */
  @Test
  public void testOutOfRangeUnread() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_out_of_range", 1, 1, Map.of(2, Set.of(new TensorType(FLOAT_64, null))));
  }
}
