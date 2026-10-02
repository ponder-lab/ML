package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a {@code train_step} reached both through {@code fit} and by a direct call with the
 * user's own batch sees the inputs of both calls (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): the step's data points both to the
 * data {@code fit} packs and to the user's tuple, and neither call's components are dropped.
 */
public class TestFitBothWays extends AbstractTensorTest {

  /** The step's inputs are the union of the two calls' inputs. */
  @Test
  public void testBothWays() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_both_ways.py",
        "consume_both",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 4), TensorType.of(FLOAT_32, 5, 6))));
  }
}
