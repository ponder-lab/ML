package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the gradient APIs that work only while a graph is being built: {@code tf.gradients}, {@code
 * tf.keras.backend.gradients} and an optimizer's {@code get_gradients} (<a
 * href="https://github.com/wala/ML/issues/1144">wala/ML#1144</a>). Each is given a list of one
 * scalar {@code float32} variable, and returns a list holding its gradient.
 */
public class TestGraphGradients extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_graph_gradients.py";

  /** {@code optimizer.get_gradients(loss, [v])[0]} is the variable's gradient. */
  @Test
  public void testGetGradients() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_get_gradients", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** {@code tf.gradients(loss, [v])[0]} is the variable's gradient. */
  @Test
  public void testGradients() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_gradients", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** {@code tf.keras.backend.gradients(loss, [v])[0]} is the variable's gradient. */
  @Test
  public void testBackendGradients() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_backend_gradients", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }
}
