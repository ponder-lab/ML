package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a value with several creators, here a parameter whose node merges several callers, is
 * typed by all of them rather than by the first one a walk reaches (<a
 * href="https://github.com/wala/ML/issues/1009">wala/ML #1009</a>). Each fixture function is nested
 * five calls below the script, deeper than the call strings reach, so its callers share one node.
 */
public class TestMergedCallers extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_merged_callers.py";

  /**
   * Two callers pass slices of different lengths, both of which resolve, so the layer's output
   * holds both shapes.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBothCallersResolve()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_both",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 5, 4), TensorType.of(FLOAT_32, 7, 4))));
  }

  /**
   * One caller passes a slice, which resolves, and the other a dataset batch, which does not: the
   * output's shape is unknown, not the slice caller's shape alone. Its dtype is known from the
   * layer.
   *
   * <p>TODO: Expect {@code (256, 4)} and {@code (5, 4)} once dataset iteration yields an element
   * the generators can read (<a href="https://github.com/wala/ML/issues/1010">wala/ML #1010</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnresolvedCallerMakesShapeUnknown()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
