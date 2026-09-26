package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for a helper dispatched on a string argument: a call site in a decidably dead arm
 * contributes no argument to its callee's parameters.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestModeDispatch extends AbstractTensorTest {

  private static final String FILE = "tf2_test_mode_dispatch.py";

  /**
   * The projection arm's helper sees the float32 hidden states of the caller that selects {@code
   * mode="projection"} alone; the embedding-mode caller's integer ids used to reach it through the
   * dead {@code elif} arm, since the dataflow union has no branch sensitivity.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testProjectionArmSeesOnlyProjectionInputs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_projection_input",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 8))));
  }

  /**
   * The embedding arm's helper sees the integer ids alone: the projection-mode caller's hidden
   * states no longer reach it through the dead {@code if} arm.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEmbeddingArmSeesOnlyEmbeddingInputs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_embedding_input", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /**
   * The control: a context whose {@code mode} depends on an unresolvable flag binds it to two
   * constants, so neither arm is decidably dead and the projection helper keeps this caller's
   * hidden states. The fold acts only on a guard whose constant is the sole binding in its context;
   * anything else is kept. (A literal flag would fold the phi to one constant and decide the arm,
   * which is the fold being precise, not the control.)
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTwoConstantsKeepBothArms()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_undecided_projection_input",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 8))));
  }

  /**
   * A three-arm dispatch: the third arm's call sits under an {@code if} and an {@code elif}, so the
   * walk from its block climbs two branch blocks. For the caller that selects the default mode both
   * edges fold to not taken and the helper receives nothing from it; the helper reads the float32
   * hidden states of the caller that selects the third mode alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testThirdArmUnderTwoGuards()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE, "consume_third_arm_input", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 8))));
  }
}
