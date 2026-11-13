package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A parameter takes the dtype an operation beside a tensor of another dtype imposes only where
 * TensorFlow converts the argument, one that is no tensor; a parameter fed a TensorFlow tensor
 * keeps its dtype, since such an operation raises when it runs (<a
 * href="https://github.com/wala/ML/issues/1253">wala/ML#1253</a>).
 */
public class TestCoerceTensorArgument extends AbstractTensorTest {

  private static final String FILE = "tf2_test_coerce_tensor_argument.py";

  /**
   * A parameter fed a {@code float32} tensor, multiplied by an {@code int32} one on an arm the run
   * does not take, keeps {@code float32}.
   */
  @Test
  public void testTensor() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale_tensor", 1, 3, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * A parameter fed a {@code float64} array, multiplied by a {@code float32} tensor, takes {@code
   * float32}, the dtype TensorFlow converts it to, the control.
   */
  @Test
  public void testArray() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale_array", 1, 3, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * A parameter fed a {@code float32} tensor through another function's parameter, multiplied by an
   * {@code int32} one on an arm the run does not take, reads both: the argument's origin is the
   * relaying parameter's, which does not say whether it is a tensor.
   */
  @Test
  public void testRelayedTensor() throws ClassHierarchyException, CancelException, IOException {
    // TODO: wala/ML#1253: TENSOR_2_FLOAT32 alone once a relay is judged by what was fed.
    test(FILE, "scale_relayed", 1, 3, Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_2_INT32)));
  }

  /**
   * A parameter fed a {@code float64} array through another function's parameter, multiplied by a
   * {@code float32} tensor, reads both its fed dtype and the one TensorFlow converts it to, for the
   * same reason.
   */
  @Test
  public void testRelayedArray() throws ClassHierarchyException, CancelException, IOException {
    // TODO: wala/ML#1253: TENSOR_2_FLOAT32 alone once a relay is judged by what was fed.
    test(FILE, "scale_array_relayed", 1, 3, Map.of(2, Set.of(TENSOR_2_FLOAT64, TENSOR_2_FLOAT32)));
  }
}
