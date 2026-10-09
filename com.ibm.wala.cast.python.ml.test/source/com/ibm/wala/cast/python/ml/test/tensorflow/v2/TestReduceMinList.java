package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code tf.math.reduce_min} over a list literal of int32 scalar tensors (wala/ML#1027): a random
 * center, its difference against a {@code tf.shape} element bound through a tuple unpacking and a
 * parameter, and a bound; and the same with the bound floor-divided by a tensor. Each reads int32.
 * The shape element reached the difference with no dtype: unpacked through a tuple, its field held
 * nothing in the heap and the literal had no generator, so the difference's allocation, read by the
 * reduction through the list, carried an unknown dtype into the result.
 */
public class TestReduceMinList extends AbstractTensorTest {

  /**
   * The first list: the center, the difference against the shape element, and the bound.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReduceMinOverList() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_reduce_min_list.py",
        "consume_min",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * The second list, whose bound is floor-divided by the first size.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReduceMinOverListWithFloorDivision()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_reduce_min_list.py",
        "consume_min_div",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32))));
  }
}
