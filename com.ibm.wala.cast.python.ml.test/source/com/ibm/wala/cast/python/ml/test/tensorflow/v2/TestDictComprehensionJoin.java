package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A key read of a dict built by a comprehension over a dict literal's {@code items()}, consumed by
 * an operation that reads its operand's points-to set.
 */
public class TestDictComprehensionJoin extends AbstractTensorTest {

  /**
   * The comprehension's dict is allocated (<a
   * href="https://github.com/wala/ML/issues/1075">wala/ML#1075</a>), but its key variable and its
   * value variable each point to every item's at once, so the write stores each key's slice under
   * every key, and the read of {@code "audios"} holds both slices. The consumer, {@code
   * tf.expand_dims}, reads its operand's points-to set and pairs every shape with every dtype, so
   * two of the four types are pairs no key has, such as the {@code "audios"} shape with the {@code
   * "ids"} dtype. At run time {@code audios_sink} receives {@code (2, 3, 1) float32}.
   *
   * <p>TODO: Expect {@code (2, 3, 1) float32} alone once <a
   * href="https://github.com/wala/ML/issues/1077">wala/ML#1077</a> is fixed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyReadIsTheJoin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dict_comprehension_join.py",
        "audios_sink",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3, 1),
                TensorType.of(FLOAT_32, 2, 5, 1),
                TensorType.of(INT_32, 2, 3, 1),
                TensorType.of(INT_32, 2, 5, 1))));
  }
}
