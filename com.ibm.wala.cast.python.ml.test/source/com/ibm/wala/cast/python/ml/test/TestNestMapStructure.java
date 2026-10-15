package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import org.junit.Test;

/**
 * A function passed to {@code tf.nest.map_structure}.
 *
 * <p>The function is analyzed only if the call graph has a node for it, and a client reading its
 * body, for example for an eager-only call, finds nothing otherwise. So this asserts on the call
 * graph's nodes.
 */
public class TestNestMapStructure extends TestPythonMLCallGraphShape {

  /**
   * {@code tf.nest} has no summary, so {@code map_structure} has no target and the function passed
   * to it gets no node, while the function making the call does.
   *
   * <p>TODO: Expect a node for {@code to_numpy} once <a
   * href="https://github.com/wala/ML/issues/1067">wala/ML#1067</a> is fixed.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testMappedFunctionHasNoNode() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_nest_map_structure.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    assertTrue("The calling function has no node.", nodes(cg, "predict") > 0);
    assertEquals("The mapped function has a node.", 0, nodes(cg, "to_numpy"));
  }

  /**
   * The number of nodes for a named function.
   *
   * @param cg The call graph.
   * @param function The function's name.
   * @return The number of the function's context-sensitive nodes.
   */
  private static int nodes(CallGraph cg, String function) {
    int ret = 0;
    for (CGNode node : cg)
      if (node.getMethod().getSignature().contains("." + function + ".")) ret++;
    return ret;
  }
}
