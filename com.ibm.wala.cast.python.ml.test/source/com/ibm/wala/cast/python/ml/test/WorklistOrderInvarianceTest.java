package com.ibm.wala.cast.python.ml.test;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.test.categories.WholeProjectFixtures;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.TestCorpusFixtures;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;
import org.junit.experimental.categories.Category;

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
@Category(WholeProjectFixtures.class)
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

  /**
   * Runs the NLPGNN generation analysis under a cycle-order shuffle seed; the run must satisfy the
   * same assertions as the unperturbed one (wala/ML#756). Reversed seeds permute only the demand
   * roots, and provably do not exercise the cycle-internal iteration orders (worklist polls,
   * dependent re-enqueues) whose identity-hash-seeded variation lets settled states differ with JVM
   * run history; the {@code ariadne.typeResolution.shuffleCycles} knob perturbs all of them
   * deterministically. Seed 11 is the recorded wala/ML#753 reproducer: it flips the BERT
   * encoder-loop {@code Transformer.call} anchor, whose exact-set pin therefore stays out of the
   * suite until that issue is fixed; the generation anchor asserted here is stable under every seed
   * tried and guards the invariance property for the acyclic region and the canonicalized cycle
   * residue.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCycleOrderInvariance()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    System.setProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY, "11");
    try {
      new TestCorpusFixtures().runNlpgnnFullGeneration();
    } finally {
      System.clearProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY);
    }
  }
}
