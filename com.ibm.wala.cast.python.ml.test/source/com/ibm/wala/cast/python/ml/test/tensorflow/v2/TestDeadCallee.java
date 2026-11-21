package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for a function that only a call in an arm a constant guard never takes reaches (<a
 * href="https://github.com/wala/ML/issues/1315">wala/ML#1315</a>). The function does not run in
 * that context, so a call it makes passes nothing on. Each fixture method unpacks a member that the
 * context whose guard is false does not hold, so an operation on it there has an unknown type.
 */
public class TestDeadCallee extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dead_callee.py";

  /** The function the arm calls passes its operation's result to another function. */
  @Test
  public void testHelper() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_helper", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The function the arm calls calls a second one, which passes its result on. */
  @Test
  public void testChain() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_chain", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A model that is called for inference with two inputs and for training with three calls a method
   * of an object it holds under {@code if training:}. The method has a node for each call of the
   * model.
   */
  @Test
  public void testModelCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_model_call", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The arm calls a method of an object that the instance holds. */
  @Test
  public void testMethod() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_method", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The control: the arm calls a method named {@code build}, which has one node for each instance,
   * so the call whose guard is true runs the node that the call whose guard is false also reaches.
   */
  @Test
  public void testBuildMethodOfOneNodeRuns()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_build", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The control: that one node of the method named {@code build} still receives the argument of the
   * call whose guard is true, and passes it on as it is.
   */
  @Test
  public void testBuildMethodOfOneNodeReceivesItsArgument()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "into_build", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The arm calls two functions that call each other, which the arm alone reaches. */
  @Test
  public void testRecursion() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_recursion", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A function that no context runs receives no tensor. */
  @Test
  public void testNeverRuns() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "never", 0, 0);
  }

  /**
   * A function that no context runs passes {@code None} and its operation's result to a function
   * that a live call passes tensors. The parameters read the live call's tensors alone.
   */
  @Test
  public void testNoneFromDeadCaller()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE, "scaled", 2, 2, Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The control: a call in the arm and a call below it reach the one node of a method named {@code
   * build}, which the call below runs.
   */
  @Test
  public void testSharedCalleeRuns() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "through_shared", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The control: a subscript that calls {@code __getitem__} has no invoke instruction in the method
   * that holds it, and the method has an arm that never runs. The call runs, so the call that
   * {@code __getitem__} makes passes its argument.
   */
  @Test
  public void testSubscriptCallRuns() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_subscript_method", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
    test(FILE, "through_subscript", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
