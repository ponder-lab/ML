package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
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
  }
}
