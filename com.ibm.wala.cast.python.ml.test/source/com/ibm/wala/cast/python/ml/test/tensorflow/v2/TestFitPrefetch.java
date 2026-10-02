package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a model fit on a dataset whose pipeline ends in a pass-through transformation has its
 * step's inputs read through the transformations to the batched element (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): the slot's generator is the
 * pass-through's, which forwards every component read to its receiver, so the packed data's reading
 * needs no unwrapping of it. A guard of that forwarding, not of a mechanism of its own.
 */
public class TestFitPrefetch extends AbstractTensorTest {

  /** The step's inputs are the batched element's first component. */
  @Test
  public void testPrefetchedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_prefetch.py",
        "consume_prefetched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
