package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A variable a closure captures after its creator's last assignment to it (wala/ML#1026): the
 * closure reads that assignment alone. The creator's scope slot received every write the creator
 * made, so a parameter rebound by a cast before a lambda captured it reached the lambda's callee as
 * both the parameter and the cast.
 */
public class TestClosureCapture extends AbstractTensorTest {

  /**
   * The lambda is created after the parameter's rebinding to the int32 cast and no write follows,
   * so its callee reads the cast alone: an int32 scalar, not the float32 the parameter arrived as.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testRebindingBeforeCapture()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_closure_rebound_capture.py",
        "consume_captured",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * The control: the rebinding follows the lambda inside a loop, so a later iteration's closure
   * reads the cast and an earlier one the parameter, and both bindings reach the callee. A write
   * reachable after the closure's allocation, here through the loop's back edge, keeps the slot.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testRebindingAfterCaptureInLoop()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_closure_rebound_capture.py",
        "consume_loop_captured",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32), TensorType.of(INT_32))));
  }

  /**
   * The control on the creator itself: the capture rule changes what the closure reads, never the
   * creator's parameter, which stays the float32 scalar its callers pass.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCreatorParameterUnchanged()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_closure_rebound_capture.py",
        "erase",
        2,
        15,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 8, 3)), 3, Set.of(TensorType.of(FLOAT_32))));
  }
}
