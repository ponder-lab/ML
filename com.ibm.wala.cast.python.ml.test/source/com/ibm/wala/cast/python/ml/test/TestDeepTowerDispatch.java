package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import org.junit.Test;

/**
 * A layer at the bottom of a tower deeper than the receiver-context depth cap is still built (<a
 * href="https://github.com/wala/ML/issues/1007">wala/ML#1007</a>). Ten wrapper layers each build
 * their inner layer lazily on the first call, and the innermost layer creates a weight. Past the
 * cap the selector degraded a trampoline to its caller's context, whose receiver is the caller's
 * instance; the trampoline's callee object is filtered to its context's receiver, so the degraded
 * trampoline dispatched nothing and the tower below it was never analysed: the innermost layer's
 * {@code build} had no node at all. Keyed on the dispatched receiver alone past the cap, it has one
 * node per receiver chain reaching it, each dispatching its body.
 */
public class TestDeepTowerDispatch extends TestPythonMLCallGraphShape {

  @Test
  public void testInnermostBuildDispatchesPastTheDepthCap() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_deep_wrapper_tower.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    int trampolines = 0;
    int undispatched = 0;
    int bodies = 0;
    for (CGNode node : cg) {
      String signature = node.getMethod().getSignature();
      if (signature.contains("Leaf.build.trampoline")) {
        trampolines++;
        if (!cg.getSuccNodes(node).hasNext()) undispatched++;
      } else if (signature.contains("Leaf.build.do")) bodies++;
    }
    assertTrue("The innermost layer's build has no trampoline node.", trampolines > 0);
    assertEquals("Build trampolines dispatching nothing.", 0, undispatched);
    assertTrue("The innermost layer's build body has no node.", bodies > 0);
  }
}
