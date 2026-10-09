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
 * The element of an array, bound by iterating a 2-D array, by a constant index, or by a
 * loop-carried index (wala/ML#1009): an array of the receiver's dtype one rank down, whose slices
 * and the arithmetic over them allocate as the receiver's do. The float literal beside the integral
 * row's slices promotes to {@code float64} as NumPy does; a row whose element reads nothing made
 * every value downstream of it empty, and the literal then read beside an unresolved operand.
 *
 * <p>The arithmetic pins read the dtype the fixture asserts and an unresolved extent where the
 * fixture asserts {@code (4,)}: the start-only slice {@code bc[2:]} degrades its axis under the
 * leading-axis rule, which folds a stop bound but not a start (wala/ML#841), and the concatenation
 * carries the degraded axis. Folding the start is a separate unit; these pins move to {@code (4,)}
 * with it.
 */
public class TestNdarrayElements extends AbstractTensorTest {

  /**
   * The row bound by iterating the array: the receiver's dtype, one rank down.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testIteratedRow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_element.py",
        "consume_row",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 5))));
  }

  /**
   * Arithmetic over the slices of the iterated row, concatenated and scaled by a float literal:
   * {@code float64}, the promotion of the integral slices.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testIteratedRowArithmetic()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_element.py",
        "consume_loop",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, List.of(UnresolvedDim.INSTANCE)))));
  }

  /**
   * The same arithmetic over the row bound by a constant index.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testIndexedRowArithmetic()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_element.py",
        "consume_index",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, List.of(UnresolvedDim.INSTANCE)))));
  }

  /**
   * The same arithmetic over the row bound by a loop-carried index.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLoopIndexedRowArithmetic()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_element.py",
        "consume_loop_index",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, List.of(UnresolvedDim.INSTANCE)))));
  }
}
