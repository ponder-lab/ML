package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the function {@code tf.map_fn} is given is called with an element of {@code elems},
 * {@code elems} without its leading dimension, so its parameter is typed from the map (<a
 * href="https://github.com/wala/ML/issues/942">wala/ML#942</a>). Before, the summary returned
 * {@code elems} without calling the function, and a function called only through {@code map_fn} had
 * no call-graph node.
 */
public class TestMapFnCallback extends AbstractTensorTest {

  private static final String FILE = "tf2_test_map_fn_callback.py";

  /**
   * A function called only by {@code map_fn}, over rows of four {@code float32} values cast from
   * {@code int32}.
   */
  @Test
  public void testCallbackOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "unpack", 1, 2, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * A function called by {@code map_fn} over rows of three {@code int32} values and directly with
   * three {@code float32} values holds both types.
   */
  @Test
  public void testCallbackAndDirectCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "scale", 1, 2, Map.of(2, Set.of(TENSOR_3_INT32, TensorType.of(FLOAT_32, 3))));
  }
}
