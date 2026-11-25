package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/** A call of {@code tf.keras.layers.Add} on a list of tensors reads as their sum (wala/ML#1327). */
public class TestKerasAddLayer extends AbstractTensorTest {

  private static final String FILE = "tf2_test_keras_add_layer.py";

  /** An {@code Add} layer called on two tensors of one shape gives that shape. */
  @Test
  public void testAddLayerCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_added", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A residual connection a layer writes with an {@code Add} layer it holds gives its input's type.
   */
  @Test
  public void testResidualAddLayer() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_summed", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * An {@code Add} layer broadcasts its inputs, so a {@code (2, 1)} input before a {@code (2, 3)}
   * one gives {@code (2, 3)}.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1329; remove {@code expected = AssertionError.class}
   * once it is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testAddLayerBroadcast() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_broadcast", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * An {@code Add} layer casts floating inputs to its compute dtype, so two float64 inputs give
   * float32.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1329; remove {@code expected = AssertionError.class}
   * once it is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testAddLayerAutocast() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_autocast", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
