package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a tensor's {@code numpy()} returns an array of the tensor's shape and dtype, and that
 * the bound method {@code t.numpy} is no tensor (wala/ML#1105). Before, the call had no type, since
 * a tensor allocation carries no method summaries, and the attribute read was typed as an element
 * of the tensor.
 */
public class TestTensorNumpyMethod extends AbstractTensorTest {

  private static final String FILE = "tf2_test_tensor_numpy_method.py";

  /** The array {@code numpy()} returns has the cast tensor's shape and dtype. */
  @Test
  public void testNumpyResult() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_array", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 2))));
  }

  /** A row of that array is an element of it. */
  @Test
  public void testNumpyResultRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /** The bound method {@code t.numpy}, passed without being called, is no tensor. */
  @Test
  public void testNumpyMethodNotATensor()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_method", 0, 0);
  }
}
