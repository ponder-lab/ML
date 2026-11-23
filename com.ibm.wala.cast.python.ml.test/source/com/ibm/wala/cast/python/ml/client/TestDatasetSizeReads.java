package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeFalse;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.function.BooleanSupplier;
import org.junit.Test;

/**
 * Tests that a read of a dataset's sizes computes each upstream dataset's sizes once while an
 * analysis resolves, not once per path through the datasets it may be made from (<a
 * href="https://github.com/wala/ML/issues/1272">wala/ML#1272</a>), also where the read is made
 * while a generator guard is in progress (<a
 * href="https://github.com/wala/ML/issues/1299">wala/ML#1299</a>).
 */
public class TestDatasetSizeReads extends AbstractTensorTest {

  /**
   * Twelve transformations, each applied on a condition, so each one's receiver is the result of
   * any of the earlier ones.
   */
  private static final String FILE = "tf2_test_dataset_optional_steps.py";

  /**
   * Nine transformations, each applied on a condition, whose batch's element is read by a function
   * that branches on a rank.
   */
  private static final String RANK_BRANCH_FILE = "tf2_test_dataset_optional_steps_rank_branch.py";

  /** The batch of four, of eight elements whichever transformations apply, has no partial batch. */
  @Test
  public void testBatch() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The analysis computes a dataset's sizes 866 times for this fixture: once for each of its 14
   * datasets, the source, the twelve transformations and the batch, and once for each of the 852
   * views of the batch over one receiver. Computed at every read, they were computed 3,464,886
   * times, each dataset once per path to it. The check mode recomputes every kept value it returns,
   * so the count says nothing under it.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSizesNotComputedPerPath() throws Exception {
    assumeFalse(Boolean.getBoolean(WorklistTypeResolver.CHECK_TRACKED_PROPERTY));
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    long computations = DatasetGenerator.sizeComputations(builder);
    assertTrue("The analysis reads a dataset's sizes.", computations > 0);
    assertTrue(
        "The analysis computed a dataset's sizes " + computations + " times.",
        computations <= 2000);
  }

  /**
   * The function that branches on the rank of the batch's element returns its rank-two arm alone,
   * whichever way the sizes are read: the batch of four, of eight elements, has two axes.
   */
  @Test
  public void testRankBranch() throws ClassHierarchyException, CancelException, IOException {
    test(RANK_BRANCH_FILE, "consume", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5, 2))));
  }

  /**
   * The analysis computes a dataset's sizes 859 times for the fixture whose element is read under a
   * rank fold. Computed at every read made while the fold is in progress, they were computed
   * 138,871 times, each dataset once per path to it. The check mode recomputes every kept value it
   * returns, so the count says nothing under it.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSizesNotComputedPerPathUnderGuard() throws Exception {
    assumeFalse(Boolean.getBoolean(WorklistTypeResolver.CHECK_TRACKED_PROPERTY));
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), RANK_BRANCH_FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    long computations = DatasetGenerator.sizeComputations(builder);
    assertTrue("The analysis reads a dataset's sizes.", computations > 0);
    assertTrue(
        "The analysis computed a dataset's sizes " + computations + " times.",
        computations <= 2000);
  }

  /** With no guard in progress, a computation in which no guard decided answers for any frame. */
  @Test
  public void testAnsweredForAnyFrame() {
    BooleanSupplier check = TensorGenerator.answeredForAnyFrame();
    assertTrue(check.getAsBoolean());
  }

  /**
   * With no guard in progress at the start, a guard's decision is the computation's own: every
   * frame it read was opened by the computation.
   */
  @Test
  public void testOwnGuardDecisionAnswersForAnyFrame() {
    BooleanSupplier check = TensorGenerator.answeredForAnyFrame();
    TensorGenerator.noteGuardDecision();
    assertTrue(check.getAsBoolean());
  }

  /** Inside a guard, a computation in which no guard decided answers for any frame. */
  @Test
  public void testUndecidedUnderGuardAnswersForAnyFrame() {
    assertTrue(
        TensorGenerator.underGuardForTest(
            () -> {
              BooleanSupplier check = TensorGenerator.answeredForAnyFrame();
              return check.getAsBoolean();
            }));
  }

  /**
   * Inside a guard, a computation past a guard's decision may answer for the frame that was open
   * around it, also once that frame has closed.
   */
  @Test
  public void testDecidedUnderGuardAnswersForItsFrame() {
    BooleanSupplier check =
        TensorGenerator.underGuardForTest(
            () -> {
              BooleanSupplier made = TensorGenerator.answeredForAnyFrame();
              TensorGenerator.noteGuardDecision();
              assertFalse(made.getAsBoolean());
              return made;
            });
    assertFalse(check.getAsBoolean());
  }
}
