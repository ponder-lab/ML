package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@code tf.keras.utils.unpack_x_y_sample_weight} outside {@code fit} (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): on a dataset element read in a
 * loop, whose components are the element's; on a tuple literal, whose components are the literal's;
 * and on a bare tensor, which is itself the inputs.
 */
public class TestUnpackForms extends AbstractTensorTest {

  private static final String FILE = "tf2_test_unpack_forms.py";

  /** A tuple dataset's element unpacks to exactly its first component. */
  @Test
  public void testLoopInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** A tuple dataset's element unpacks to exactly its second component. */
  @Test
  public void testLoopTargets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_targets", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A tuple literal unpacks to its components. */
  @Test
  public void testLiteralInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_literal_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5, 1))));
  }

  /**
   * A bare tensor unpacks to itself at index 0. The value also reaches the sink through the heap
   * edge the pointer analysis gives the result's slot, so this pin guards the value, not the arm
   * that computes it.
   */
  @Test
  public void testBare() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bare", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6, 2))));
  }
}
