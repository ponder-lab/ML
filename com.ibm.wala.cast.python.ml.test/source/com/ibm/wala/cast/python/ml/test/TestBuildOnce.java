package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.ContextKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.util.collections.HashMapFactory;
import java.util.Map;
import org.junit.Test;

/**
 * A Keras layer builds once per instance: {@code Layer.__call__} runs {@code build} only while
 * {@code self.built} is false, and an explicit {@code layer.build(input_shape)} sets it, so the
 * lazy build a layer call injects and the program's explicit call reach one build of the instance.
 * The layer tower of {@code tf2_test_lazy_build_tower.py} builds every sublayer explicitly from its
 * parent's {@code build} and calls it from its parent's {@code call}; with a build trampoline keyed
 * on its caller and site, each receiver had one node per build site per receiver chain, and every
 * context beneath the tower doubled with it. Keyed on the receiver alone, a build trampoline has
 * one node per instance (<a href="https://github.com/wala/ML/issues/1013">wala/ML#1013</a>).
 */
public class TestBuildOnce extends TestPythonMLCallGraphShape {

  @Test
  public void testOneBuildTrampolinePerReceiver() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_lazy_build_tower.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    Map<InstanceKey, Integer> nodesPerReceiver = HashMapFactory.make();
    for (CGNode node : cg)
      if (node.getMethod().getSignature().contains("Minkowski.build.trampoline")) {
        InstanceKey receiver = (InstanceKey) node.getContext().get(ContextKey.RECEIVER);
        assertNotNull("A build trampoline node names no receiver: " + node, receiver);
        nodesPerReceiver.merge(receiver, 1, Integer::sum);
      }
    assertTrue("No build trampoline for the innermost layer.", !nodesPerReceiver.isEmpty());
    for (Map.Entry<InstanceKey, Integer> e : nodesPerReceiver.entrySet())
      assertEquals(
          "Build trampoline nodes for receiver " + e.getKey() + ": " + e.getValue(),
          1,
          (int) e.getValue());
  }
}
