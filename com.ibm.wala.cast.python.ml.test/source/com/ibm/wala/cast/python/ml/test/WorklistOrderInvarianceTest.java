package com.ibm.wala.cast.python.ml.test;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;

/**
 * Order-independence guard for the worklist engine (wala/ML#365, Phase 2): the whole-project NLPGNN
 * generation analysis must satisfy its type assertions under the engine regardless of the seeding
 * order. The round-based resolution cannot make this guarantee (its cycle guards return
 * stack-dependent approximations, so the first reader fixes each round's cached value); the
 * engine's chaotic iteration to a fixpoint can, and this test keeps it honest by running the same
 * analysis with the seeding order forward and reversed.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class WorklistOrderInvarianceTest {

  /**
   * Runs the NLPGNN generation analysis under the engine with both seeding orders; both runs must
   * satisfy the same assertions.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSeedOrderInvariance()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    System.setProperty("ariadne.typeResolution.worklist", "true");
    try {
      new TestTensorflow2Model().runNlpgnnFullGeneration();
      System.setProperty("ariadne.typeResolution.reverseSeeds", "true");
      new TestTensorflow2Model().runNlpgnnFullGeneration();
    } finally {
      System.clearProperty("ariadne.typeResolution.worklist");
      System.clearProperty("ariadne.typeResolution.reverseSeeds");
    }
  }
}
