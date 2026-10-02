package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a model rebuilt from its configuration keeps its layers' constraints (<a
 * href="https://github.com/wala/ML/issues/996">wala/ML#996</a>): the round trip through {@code
 * serialize_keras_object}, {@code layers.deserialize}, {@code constraints.serialize} and {@code
 * constraints.get} is modeled as the identity, so training the rebuilt model applies the original
 * constraint to the weight.
 */
public class TestConfigRoundTrip extends AbstractTensorTest {

  /** The rebuilt model's nested layer applies the original constraint to its weight. */
  @Test
  public void testRebuiltConstraint() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_round_trip.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }
}
