package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a model whose inputs are a tensor on one path and a dataset on the other, through one
 * {@code fit} call, gives the step both inputs (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): the one packed data's slot holds a
 * tensor and a dataset, and neither is dropped.
 */
public class TestFitMixedSites extends AbstractTensorTest {

  /** The step's inputs are the tensor path's value and the dataset path's element component. */
  @Test
  public void testMixedSites() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_mixed_sites.py",
        "consume_mixed",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 4), TensorType.of(FLOAT_32, 2, 4))));
  }
}
