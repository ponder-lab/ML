package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the tensors read back from a {@code tf.TensorArray}, which have the dtype the array was
 * built with. Their shape is not read: the static shape TensorFlow gives a stacked array depends on
 * its size and element shape, so each reads an unknown shape.
 */
public class TestTensorArray extends AbstractTensorTest {

  private static final String FILE = "tf2_test_tensor_array.py";

  /**
   * An array built with a positional dtype, written to and stacked.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testStacked() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stacked", 1, 1, Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * An element read from an array.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testRead() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_read", 1, 1, Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * An array filled by {@code unstack} and stacked.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnstacked() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unstacked", 1, 1, Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * {@code tf.gather_nd} of a stacked array, as a beam search reads its best index.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testGathered() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_gathered", 1, 1, Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * An array built with {@code dtype=} given by keyword.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testKeywordDType() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
