package com.ibm.wala.cast.python.ml.test;

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
   * {@code map_structure} calls the function it is given, so the function has a node, as the
   * function making the call does (<a
   * href="https://github.com/wala/ML/issues/1067">wala/ML#1067</a>). Before, {@code tf.nest} had no
   * summary, the call had no target, and the function had no node.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testMappedFunctionHasNode() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_nest_map_structure.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    assertTrue("The calling function has no node.", nodes(cg, "predict") > 0);
    assertTrue("The mapped function has no node.", nodes(cg, "to_numpy") > 0);
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
