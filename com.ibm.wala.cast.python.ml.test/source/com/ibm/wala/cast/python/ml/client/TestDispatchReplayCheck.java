package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import java.io.File;
import java.util.Collections;
import org.junit.Test;

/**
 * Tests that the dispatches a walk keeps for a cycle of values, replayed, give the generators the
 * dispatches computed afresh give (<a
 * href="https://github.com/wala/ML/issues/1274">wala/ML#1274</a>). The class has one test, since it
 * turns on the factory's check mode for its JVM.
 */
public class TestDispatchReplayCheck extends AbstractTensorTest {

  /** Five mutually recursive functions, each returning a call to one of the next two. */
  private static final String FILE = "tf2_test_dispatch_return_cycle_small.py";

  /**
   * The analysis replays dispatches of the cycle's values, and under the factory's check mode,
   * which computes each replayed dispatch again from the path replaying it, none differs.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testReplaysAgreeWithFreshDispatches() throws Exception {
    String before = System.getProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY);
    System.setProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY, "true");
    try {
      long checks = TensorGeneratorFactory.replayChecks();
      long differences = TensorGeneratorFactory.replayDifferences();
      PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      builder.makeCallGraph(builder.getOptions());
      engine.performAnalysis(builder);
      assertTrue(
          "The analysis replays a dispatch of the cycle.",
          TensorGeneratorFactory.replayChecks() > checks);
      assertEquals(differences, TensorGeneratorFactory.replayDifferences());
    } finally {
      if (before == null) System.clearProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY);
      else System.setProperty(TensorGeneratorFactory.CHECK_REPLAYS_PROPERTY, before);
    }
  }
}
