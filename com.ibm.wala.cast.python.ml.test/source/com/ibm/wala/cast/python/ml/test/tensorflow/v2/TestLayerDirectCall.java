package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a layer's {@code call} invoked directly, {@code layer.call(x)}, receives its argument
 * as passed, while applying the layer, {@code layer(x)}, casts a floating-point argument to the
 * layer's compute dtype (wala/ML#1114, wala/ML#821). Before, the input of every {@code call} body
 * was cast, so a direct invocation was modeled as an application.
 */
public class TestLayerDirectCall extends AbstractTensorTest {

  private static final String FILE = "tf2_test_layer_direct_call_dtype.py";

  /**
   * Applying the layer to a {@code float64} tensor gives its {@code call} a {@code float32} one.
   */
  @Test
  public void testAppliedCast() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "applied_inside", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * Invoking {@code call} directly with a {@code float64} tensor passes it on as {@code float64}.
   */
  @Test
  public void testDirectNoCast() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "direct_inside", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 2))));
  }
}
