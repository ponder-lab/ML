package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a φ whose governing branch folds drops its dead arm's flow, whatever the number of
 * blocks the arms span (wala/ML#970).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestPhiArmFeasibility extends AbstractTensorTest {

  private static final String FILE = "tf2_test_phi_arm_chain.py";

  private static final TensorType TENSOR_2_6_FLOAT32 = TensorType.of(FLOAT_32, 2, 6);

  /**
   * An {@code if self.concat is True:} guard over a field bound to {@code True} at construction,
   * whose arms each make two calls: the live arm's {@code (2, 6)} result alone reaches the sink,
   * and the dead {@code reduce_mean} arm's {@code (2, 2)} does not. Each call ends a block, so the
   * arm reaches the merge through a chain of blocks; the feasibility check had looked one block up
   * from the arm's predecessor and so never found the branch, leaving both arms in the φ.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDeadArmPrunedAcrossBlockChain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_multi", 1, 1, Map.of(2, Set.of(TENSOR_2_6_FLOAT32)));
  }

  /**
   * The same guard over arms of one call each: even then an arm spans more than one block (the
   * call's operands are built in blocks of their own), so the one-hop check missed this form too.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDeadArmPrunedSingleBlock()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_single", 1, 1, Map.of(2, Set.of(TENSOR_2_6_FLOAT32)));
  }

  /**
   * A decided guard inside a loop: the loop header's merge, reached from below through the guard's
   * own merge, stays undecided, so both the value that enters the loop ({@code (2, 2)}, the result
   * when the loop runs zero times) and the one the loop writes ({@code (2, 6)}) reach the sink, as
   * the two calls' runtime values do. Pins that the chain walk stops at a block with two
   * predecessors rather than walking through the back edge.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLoopHeaderStaysUndecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_6_FLOAT32, TENSOR_2_2_FLOAT32)));
  }
}
