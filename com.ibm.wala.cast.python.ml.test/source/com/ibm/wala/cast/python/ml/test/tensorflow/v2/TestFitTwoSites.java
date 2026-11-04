package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a model fit at two sites on datasets of different element types gives its step both
 * datasets' inputs (<a href="https://github.com/wala/ML/issues/997">wala/ML#997</a>).
 */
public class TestFitTwoSites extends AbstractTensorTest {

  /** The step's inputs are either dataset's element component. */
  @Test
  public void testTwoSites() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_two_sites.py",
        "consume_x",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), TensorType.of(INT_32, 2, 3))));
  }
}
