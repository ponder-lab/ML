package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
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
 * A subscript by an integer index drops the indexed axis whatever the index's value, so an index
 * with no constant value, such as a loop counter or one computed at run time, drops it as a
 * constant does (<a href="https://github.com/wala/ML/issues/1247">wala/ML#1247</a>).
 */
public class TestSubscriptIntegerIndex extends AbstractTensorTest {

  private static final String FILE = "tf2_test_subscript_integer_index.py";

  /** A leading axis of unknown extent kept beside the receiver's second, of four. */
  private static final TensorType TENSOR_UNRESOLVED_4_FLOAT32 =
      new TensorType(FLOAT_32, List.of(UnresolvedDim.INSTANCE, new NumericDim(4)));

  /** A constant index drops its axis. */
  @Test
  public void testRowConstant() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row_constant", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /** An index computed at run time drops its axis. */
  @Test
  public void testRowUnknown() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row_unknown", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * An {@code enumerate} index over a list of two elements, which has two values, drops its axis.
   */
  @Test
  public void testRowEnumerate() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row_enumerate", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /** An index passed as a parameter drops its axis. */
  @Test
  public void testRowParameter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row_parameter", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /** An index beside a bounded slice drops its axis, and the slice keeps its extent. */
  @Test
  public void testRowSliced() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row_sliced", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** An index computed at run time, alone, drops its axis. */
  @Test
  public void testElementUnknown() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_element_unknown", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /** An index computed at run time drops an array's axis. */
  @Test
  public void testArrayRowUnknown() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_array_row_unknown", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT64)));
  }

  /** An index computed at run time after a full slice drops the second axis. */
  @Test
  public void testColumnUnknown() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_column_unknown", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * An index that may be an integer or a float constant, which is not known to be an integer, so
   * the subscript is read as a slice of the leading axis.
   *
   * <p>TODO: <a href="https://github.com/wala/ML/issues/1251">wala/ML#1251</a>: {@link
   * #TENSOR_4_FLOAT32}, as the run gives.
   */
  @Test
  public void testRowMaybeFloat() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row_maybe_float", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_4_FLOAT32)));
  }

  /**
   * An index that may be an integer or a float of unknown value, read as {@link #testRowMaybeFloat}
   * reads its index.
   *
   * <p>TODO: <a href="https://github.com/wala/ML/issues/1251">wala/ML#1251</a>: {@link
   * #TENSOR_4_FLOAT32}, as the run gives.
   */
  @Test
  public void testRowMaybeFloatUnknown()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_row_maybe_float_unknown",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNRESOLVED_4_FLOAT32)));
  }
}
