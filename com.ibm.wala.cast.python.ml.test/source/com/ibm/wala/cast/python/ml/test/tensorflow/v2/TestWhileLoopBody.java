package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code tf.while_loop} calls its {@code body} with the loop variables unpacked
 * positionally, so a function reached only through a loop body gets a call-graph node and its
 * parameters read the loop variables' types (<a
 * href="https://github.com/wala/ML/issues/942">wala/ML#942</a>).
 */
public class TestWhileLoopBody extends AbstractTensorTest {

  private static final String FILE = "tf2_test_while_loop_body.py";

  /**
   * A function called from a lambda body, the loop variable at the body's second position: its
   * parameter reads the image the loop was handed.
   */
  @Test
  public void testLambdaBodyCallee() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_image", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 4, 3))));
  }

  /** A named body passed through a variable: its first parameter reads the int32 loop counter. */
  @Test
  public void testNamedBodyFirstVariable()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_index", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * The named body's third parameter reads the third loop variable, and nothing at another
   * position.
   */
  @Test
  public void testNamedBodyThirdVariable()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_boxes", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 4))));
  }
}
