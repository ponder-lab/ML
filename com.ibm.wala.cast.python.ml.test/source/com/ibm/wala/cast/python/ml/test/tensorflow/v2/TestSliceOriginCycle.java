package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A loop-carried slice on {@code tf2_test_slice_origin_cycle.py}: {@code context = context[1:]}
 * inside a {@code while} loop, so the slice's receiver is a φ of the slice's own result. A slice
 * classifies its origin through its receiver's generator (wala/ML#731), and that generator is the
 * slice again, so the classification revisited itself until the stack was exhausted and the
 * analysis died before the first type. With the receiver guarded while it is being classified, the
 * revisit reads as unresolved and the analysis completes. The witness is completion with the sink
 * read as a scalar {@code float32}: on the previous engine this test ends in a {@code
 * StackOverflowError} (wala/ML#979).
 */
public class TestSliceOriginCycle extends AbstractTensorTest {

  @Test
  public void testLoopCarriedSliceCompletes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_slice_origin_cycle.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }
}
