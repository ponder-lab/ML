package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.UNKNOWN;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code np.arange}, {@code np.pad} and {@code np.random.randint} (wala/ML#909): a pad's extent is
 * the input's extent plus its widths, folded as terms over the program's own values so that a width
 * written against the input's own length cancels to the constant the program fixes.
 */
public class TestNumpyPadArange extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_numpy_pad_arange.py";

  /**
   * {@code np.arange} in its positional and keyword forms: a constant extent from constant bounds,
   * an unresolved one from a bound the program decides at run time, rank one either way.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testArange()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_arange_stop", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 10))));
    test(FIXTURE, "consume_arange_bounds", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 8))));
    test(FIXTURE, "consume_arange_step", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
    test(FIXTURE, "consume_arange_keywords", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
    test(FIXTURE, "consume_arange_mixed", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 8))));
    test(
        FIXTURE,
        "consume_arange_unresolved",
        1,
        1,
        Map.of(2, Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE)))));
  }

  /**
   * {@code np.pad} with constant inputs and widths: an integer width, a {@code (before, after)}
   * pair, and per-axis pairs on a rank-2 input.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPadConstants()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_pad_scalar", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 9))));
    test(FIXTURE, "consume_pad_pair", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 8))));
    test(FIXTURE, "consume_pad_axes", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 4, 5))));
    test(FIXTURE, "consume_pad_single_width", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 9))));
    test(
        FIXTURE, "consume_pad_single_pair", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 5, 6))));
    // An input the chase does not read (`np.eye`) but whose shape is typed folds from the shape.
    test(
        FIXTURE, "consume_pad_typed_input", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 5, 5))));
  }

  /**
   * The cancellation: an unresolved length {@code n} padded by {@code total - n} is {@code total}
   * long, whether {@code n} arrives as an {@code arange} bound, through an elementwise rescale of
   * an {@code arange} whose own bounds cancel, or as a sized random draw's {@code size}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPadCancels()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_pad_cancels", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 10))));
    test(
        FIXTURE, "consume_pad_cancels_scaled", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 12))));
    // A sized draw whose size the program decides at run time reads unknown rank, the sized-draw
    // generator's standing rule for a supplied but unresolvable size; the pad after it still folds,
    // because the chase reads the draw's `size` argument as a term, not its emitted dimension.
    test(FIXTURE, "consume_randint", 1, 1, Map.of(2, Set.of(new TensorType(INT_64, null))));
    test(FIXTURE, "consume_pad_cancels_draw", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 15))));
    // The scalar on the left of the rescale, an arange with only a stop, and coefficients. The
    // left-scaled product's dtype followed the program when the elementwise rule was fixed
    // (wala/ML#922): an integer literal no longer decides, and the array's dtype is unknown to the
    // analysis (a runtime bound), so the product is unknown, where it used to read int32. The
    // expectation was written as knowingly divergent with the instruction to follow the program
    // rather than restore int32, and this is that instruction carried out; the extent is what this
    // test owns.
    test(FIXTURE, "consume_pad_scaled_left", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 10))));
    test(FIXTURE, "consume_pad_stop_only", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 10))));
    test(FIXTURE, "consume_pad_coefficient", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 20))));
  }

  /**
   * What the fold declines: a width that does not cancel the unresolved length, and a width the
   * program decides at run time on a constant input; the rank stays one and the extent unresolved.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPadDeclines()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_pad_open",
        1,
        1,
        Map.of(2, Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE)))));
    test(
        FIXTURE,
        "consume_pad_unknown_width",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, asList(UnresolvedDim.INSTANCE)))));
    Set<TensorType> unresolvedUnknown =
        Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE)));
    // Inputs the chase declines on: two arrays combined, an arange with a step, a float bound.
    test(FIXTURE, "consume_pad_two_arrays", 1, 1, Map.of(2, unresolvedUnknown));
    test(FIXTURE, "consume_pad_stepped_input", 1, 1, Map.of(2, unresolvedUnknown));
    test(FIXTURE, "consume_pad_float_bound", 1, 1, Map.of(2, unresolvedUnknown));
    // A typed but unresolved input keeps its rank with unresolved axes; a floor-division width
    // is an operator the fold does not express.
    test(
        FIXTURE,
        "consume_pad_typed_unresolved",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_64, asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)))));
    test(
        FIXTURE,
        "consume_pad_floordiv",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, asList(UnresolvedDim.INSTANCE)))));
  }

  /** {@code np.cumsum([0, 3, 2])} is a rank-1 int64 array of three (wala/ML#954). */
  @Test
  public void testCumsumLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cumsum_offsets.py",
        "consume_literal",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new NumericDim(3))))));
  }

  /** With a constant {@code axis} the running sum keeps the input's shape: {@code (2, 2)} int64. */
  @Test
  public void testCumsumAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cumsum_offsets.py",
        "consume_axis",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new NumericDim(2), new NumericDim(2))))));
  }

  /** An explicit {@code dtype} overrides the accumulator rule: {@code (3,)} float64. */
  @Test
  public void testCumsumDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cumsum_offsets.py",
        "consume_dtype",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, asList(new NumericDim(3))))));
  }

  /**
   * The accumulator rule: an int32 input, narrower than the platform integer, sums as int64 ({@code
   * np.cumsum(np.array([1, 2], dtype=np.int32))} is {@code (2,)} int64).
   */
  @Test
  public void testCumsumWidensNarrowInteger()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cumsum_offsets.py",
        "consume_widened",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new NumericDim(2))))));
  }

  /**
   * An unsigned narrow input sums as uint64 at run time, which {@code DType} has no constant for,
   * so the dtype reads unknown rather than an int64 a signature would assert wrongly.
   */
  @Test
  public void testCumsumUnsignedNarrowInteger()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cumsum_offsets.py",
        "consume_uint8",
        1,
        1,
        Map.of(2, Set.of(new TensorType(UNKNOWN, asList(new NumericDim(2))))));
  }

  /** A computed {@code axis} still keeps the input's shape: the axis value never matters. */
  @Test
  public void testCumsumComputedAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cumsum_offsets.py",
        "consume_computed_axis",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new NumericDim(2), new NumericDim(2))))));
  }
}
