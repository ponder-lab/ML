package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
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
}
