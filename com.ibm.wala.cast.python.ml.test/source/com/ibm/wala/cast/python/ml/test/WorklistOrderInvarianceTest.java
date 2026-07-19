package com.ibm.wala.cast.python.ml.test;

import com.ibm.wala.cast.python.ml.test.tensorflow.v2.TestCorpusFixtures;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;

/**
 * Order-independence guard for the worklist engine (wala/ML#365): the whole-project NLPGNN
 * generation analysis must satisfy its type assertions regardless of the seeding order. The retired
 * round-based resolution could not make this guarantee (its cycle guards returned stack-dependent
 * approximations, so the first reader fixed each round's cached value); the engine's chaotic
 * iteration to a fixpoint can, and this test keeps it honest by running the same analysis with the
 * seeding order forward and reversed.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class WorklistOrderInvarianceTest {

  /**
   * Runs the NLPGNN generation analysis with both seeding orders; both runs must satisfy the same
   * assertions.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSeedOrderInvariance()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    new TestCorpusFixtures().runNlpgnnFullGeneration();
    System.setProperty("ariadne.typeResolution.reverseSeeds", "true");
    try {
      new TestCorpusFixtures().runNlpgnnFullGeneration();
    } finally {
      System.clearProperty("ariadne.typeResolution.reverseSeeds");
    }
  }
}
