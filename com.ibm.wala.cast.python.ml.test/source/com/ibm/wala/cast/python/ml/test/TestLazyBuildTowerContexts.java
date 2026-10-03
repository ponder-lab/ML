package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.TrampolineReceiverContextSelector.AnchoredCallerSiteContext;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.ContextKey;
import org.junit.Test;

/**
 * A tower of lazily built layers is analysed with contexts that count receiver chains and call
 * sites, not call paths (<a href="https://github.com/wala/ML/issues/1007">wala/ML#1007</a>). The
 * model's kernel is a gate over scaled, distance-based subnets wrapping a constrained distance
 * layer, every level built on its first call, used through {@code fit}, {@code evaluate} and {@code
 * predict} before and after a rebuild from its config. Keying a context on the calling node made a
 * method's contexts its distinct call paths from a receiver root, one more factor at every fan-in
 * along the tower: the innermost layer's {@code build} trampoline had 91 nodes for one instance, of
 * which 70 dispatched nothing, since past the receiver-depth cap a trampoline inherited its
 * caller's context and its callee object was filtered to the wrong receiver. Anchored on the
 * nearest receiver-keyed node, the calling method and the site, it has 56, every one dispatching
 * its body, and the program 725 in all against 1,028. The bounds sit between the two.
 */
public class TestLazyBuildTowerContexts extends TestPythonMLCallGraphShape {

  /** The most nodes the innermost layer's {@code build} trampoline may have for this program. */
  private static final int MAX_BUILD_TRAMPOLINE_NODES = 70;

  /** The most nodes the whole program may have. */
  private static final int MAX_NODES = 900;

  @Test
  public void testTowerContextsCountReceiverChains() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_lazy_build_tower.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    int trampolines = 0;
    int undispatched = 0;
    for (CGNode node : cg)
      if (node.getMethod().getSignature().contains("Minkowski.build.trampoline")) {
        trampolines++;
        if (!cg.getSuccNodes(node).hasNext()) undispatched++;
      }
    assertTrue("No build trampoline for the innermost layer.", trampolines > 0);
    assertEquals("Build trampolines dispatching nothing.", 0, undispatched);
    assertTrue(
        "build trampoline nodes: " + trampolines + " > " + MAX_BUILD_TRAMPOLINE_NODES,
        trampolines <= MAX_BUILD_TRAMPOLINE_NODES);
    assertTrue(
        "nodes: " + cg.getNumberOfNodes() + " > " + MAX_NODES, cg.getNumberOfNodes() <= MAX_NODES);
  }

  /**
   * An anchored context answers no calling node and does answer its call site. The anchor is not
   * the calling node, and several calling nodes with the same method share one context, so a caller
   * it named would be an arbitrary representative whose own context follows solver order; a reader
   * pairing the caller and call-site keys must get nothing rather than a mismatched pair.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAnchoredContextsNameNoCaller() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_lazy_build_tower.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    int anchored = 0;
    for (CGNode node : cg) {
      Context context = node.getContext();
      if (!(context instanceof AnchoredCallerSiteContext)) continue;
      anchored++;
      assertNull("An anchored context names a calling node.", context.get(ContextKey.CALLER));
      assertNotNull("An anchored context names no call site.", context.get(ContextKey.CALLSITE));
      assertNotNull(
          "An anchored context names no anchor.",
          ((AnchoredCallerSiteContext) context).getAnchor());
    }
    assertTrue("No anchored context in the graph.", anchored > 0);
  }
}
