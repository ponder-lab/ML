package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Tests that the dispatches a walk keeps for a cycle of values, replayed, give the generators the
 * dispatches computed afresh give (<a
 * href="https://github.com/wala/ML/issues/1274">wala/ML#1274</a>), and that the check reaches the
 * dispatches that replayed one. Each test turns on the factory's check mode for its analysis.
 */
public class TestDispatchReplayCheck extends AbstractTensorTest {

  /** Five mutually recursive functions, each returning a call to one of the next two. */
  private static final String FILE = "tf2_test_dispatch_return_cycle_small.py";

  /** Two functions that call each other, one returning ones and the other zeros. */
  private static final String MUTUAL_RECURSION = "tf2_test_dispatch_mutual_recursion.py";

  /**
   * The analysis replays dispatches of the cycle's values, and under the factory's check mode,
   * which computes each replayed dispatch again from the path replaying it, none differs.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testReplaysAgreeWithFreshDispatches() throws Exception {
    assertEquals(0, differingReplays(FILE, false));
  }

  /**
   * The result of one function's call in the other replays the memoized dispatch of that function's
   * return value, which was cut, and is not cut itself. Its own replay differs from the dispatch
   * computed afresh, which the check finds by taking a dispatch that replayed an inexact one as
   * inexact.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testReplayOfReplayingDispatchIsChecked() throws Exception {
    assertEquals(1, differingReplays(MUTUAL_RECURSION, true));
  }

  /**
   * Dispatches a fixture's values under the factory's check mode.
   *
   * @param file The fixture.
   * @param everyValue Whether to dispatch every value ({@link #dispatchEveryValue}) rather than the
   *     values an analysis dispatches.
   * @return How many replays the check found to differ from the dispatch computed afresh.
   * @throws Exception On analysis failure.
   */
  private long differingReplays(String file, boolean everyValue) throws Exception {
    String before = System.getProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY);
    System.setProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY, "true");
    try {
      long checks = TensorGeneratorFactory.replayChecks();
      long differences = TensorGeneratorFactory.replayDifferences();
      PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), file);
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      builder.makeCallGraph(builder.getOptions());
      if (everyValue) dispatchEveryValue(builder);
      else engine.performAnalysis(builder);
      assertTrue(
          "The analysis replays a dispatch of the cycle.",
          TensorGeneratorFactory.replayChecks() > checks);
      return TensorGeneratorFactory.replayDifferences() - differences;
    } finally {
      if (before == null) System.clearProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY);
      else System.setProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY, before);
    }
  }

  /**
   * Dispatches every value of a call graph in the reverse of the pointer analysis's order, twice,
   * with the resolver installed so the dispatches are memoized.
   *
   * @param builder The call graph's builder.
   */
  private static void dispatchEveryValue(PythonSSAPropagationCallGraphBuilder builder) {
    WorklistTypeResolver.install(builder);
    try {
      // A snapshot: dispatch creates points-to set variables for keys it reads.
      List<PointerKey> keys = new ArrayList<>();
      builder.getPointerAnalysis().getPointerKeys().forEach(keys::add);
      Collections.reverse(keys);
      // Twice, so each memoized dispatch is replayed.
      for (int pass = 0; pass < 2; pass++)
        for (PointerKey key : keys) {
          if (builder.getPropagationSystem().isImplicit(key)) continue;
          PointsToSetVariable variable =
              builder.getPropagationSystem().findOrCreatePointsToSet(key);
          try {
            TensorGeneratorFactory.getGenerator(variable, builder);
          } catch (IllegalArgumentException e) {
            // A value the factory refuses is refused again.
          }
        }
    } finally {
      WorklistTypeResolver.uninstall(builder, false);
      TensorGenerator.clearCaches(builder);
    }
  }
}
