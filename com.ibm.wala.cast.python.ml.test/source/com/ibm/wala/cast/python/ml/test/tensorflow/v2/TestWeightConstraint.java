package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a weight constraint passed to {@code add_weight} is applied to the weight, so the
 * constraint's {@code __call__} receives the weight's type (<a
 * href="https://github.com/wala/ML/issues/996">wala/ML#996</a>). The constraint reaches {@code
 * add_weight} either through a constructor keyword stored on the layer and resolved by {@code
 * tf.keras.constraints.get}, or constructed inline at the call.
 */
public class TestWeightConstraint extends AbstractTensorTest {

  private static final String FILE = "tf2_test_weight_constraint.py";

  /** The stored constraint's {@code __call__} receives the {@code (4, 3)} weight. */
  @Test
  public void testStoredConstraint() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "NonNegNorm.__call__", 1, 10, Map.of(3, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /** The inline constraint's {@code __call__} receives the {@code (5, 2)} weight. */
  @Test
  public void testInlineConstraint() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "InlineNorm.__call__", 1, 2, Map.of(3, Set.of(TensorType.of(FLOAT_32, 5, 2))));
  }

  /** The layer with the stored constraint still computes its output. */
  @Test
  public void testStoredLayerOutput() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stored", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The layer with the inline constraint still computes its output. */
  @Test
  public void testInlineLayerOutput() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inline", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }
}
