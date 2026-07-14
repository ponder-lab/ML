package com.ibm.wala.cast.python.ml.test;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine.DepthLimitedResult;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

/**
 * Regression guards for the depth-too-short signal (<a
 * href="https://github.com/wala/ML/issues/601">wala/ML#601</a>): {@link
 * PythonTensorAnalysisEngine#getDepthLimitedResults()} reports the ⊤ (unknown-shape) tensor values
 * whose targeted-context nodes sit at a call string saturated to the configured {@code
 * targetedCfaDepth}, so a consumer can tell an insufficient depth (rather than a genuine unknown)
 * produced the ⊤ and tune the depth to the subject's fixed point.
 *
 * <p>The fixture is {@code neural_network.py}, whose accuracy path threads the input through the
 * framework ops {@code cast}/{@code reduce_mean}/{@code argmax}/{@code equal}. At the default depth
 * their call strings are truncated before reaching a distinguishing caller context, so the signal
 * is non-empty; by the model-forward depth their call strings reach the call-graph root within the
 * budget (the signal is empty), which is the subject's fixed point.
 */
public class TestDepthLimitedSignal extends TestPythonMLCallGraphShape {

  private List<DepthLimitedResult> depthLimitedResults(int targetedCfaDepth)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(targetedCfaDepth, emptyList(), "neural_network.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    return engine.getDepthLimitedResults();
  }

  /**
   * At the context-insensitive depth {@code 0} there is no call-string budget to deepen, so no ⊤ is
   * attributable to a truncated call string and the signal is empty.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmptyAtDepthZero()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(List.of(), depthLimitedResults(0));
  }

  /**
   * At the default depth the accuracy-path framework ops resolve to ⊤ at a saturated call string,
   * so the signal is non-empty; every reported value carries a ⊤ shape, sits at a node routed
   * through the targeted selector, and has a call string of exactly the configured depth.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testFiresAtDefaultDepth()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<DepthLimitedResult> results =
        depthLimitedResults(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
    assertTrue("Expected a depth-too-short signal at the default depth", results.size() > 0);
    for (DepthLimitedResult result : results) {
      assertEquals(
          PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, result.callStringLength());
      assertTrue(
          "Every reported node should be in the framework subgraph",
          result
              .node()
              .getMethod()
              .getDeclaringClass()
              .getName()
              .toString()
              .contains("tensorflow"));
    }
  }

  /**
   * By the model-forward depth every targeted call string reaches the call-graph root within the
   * budget, so the signal is empty: the subject's fixed point, past which the remaining ⊤ values
   * are genuine unknowns rather than merged-context artifacts.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmptyAtFixedPoint()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertEquals(
        List.of(), depthLimitedResults(PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH));
  }
}
