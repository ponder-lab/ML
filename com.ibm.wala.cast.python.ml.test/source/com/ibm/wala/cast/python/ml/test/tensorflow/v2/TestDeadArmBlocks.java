package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for a call anywhere in an arm a constant guard never takes (<a
 * href="https://github.com/wala/ML/issues/1316">wala/ML#1316</a>). The call passes nothing to its
 * callee, whether it stands directly under the guard, after an {@code if} statement in the arm, in
 * a loop in the arm or after one. The fixture's arm unpacks a member that the context whose guard
 * is false does not hold, so an operation on it there has an unknown type.
 */
public class TestDeadArmBlocks extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dead_arm_blocks.py";

  /** The control: a call directly under the guard. */
  @Test
  public void testUnderGuard() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "under_guard", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call after an {@code if} statement with an {@code else} in the arm. */
  @Test
  public void testAfterBranch() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "after_branch", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call after an {@code if} statement with no {@code else} in the arm. */
  @Test
  public void testAfterOneArmedBranch()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "after_one_armed_branch", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call in a {@code for} loop in the arm. */
  @Test
  public void testInForLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_for_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call after a {@code for} loop in the arm. */
  @Test
  public void testAfterForLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "after_for_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call in a {@code while} loop in the arm. */
  @Test
  public void testInWhileLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_while_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call in a loop inside a loop in the arm. */
  @Test
  public void testInNestedLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_nested_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The control: a call past the arm runs in both contexts. */
  @Test
  public void testPastArm() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "past_arm", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /** The control: a call in a loop in an arm whose guard is no constant. */
  @Test
  public void testUndecidedInLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "undecided_in_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The control: a call after an {@code if} statement in an arm whose guard is no constant. */
  @Test
  public void testUndecidedAfterBranch()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "undecided_after_branch", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The control: a call in an exception handler, which no branch leads to. */
  @Test
  public void testInHandler() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_handler", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The control: a call after a {@code try} statement whose handler runs. */
  @Test
  public void testAfterHandler() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "after_handler", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A call in a loop in an {@code else} arm that never runs passes nothing. */
  @Test
  public void testInElseArm() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_else_arm", 0, 0);
  }

  /** The control: a call after an {@code else} arm that never runs. */
  @Test
  public void testAfterElseArm() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "after_else_arm", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }
}
