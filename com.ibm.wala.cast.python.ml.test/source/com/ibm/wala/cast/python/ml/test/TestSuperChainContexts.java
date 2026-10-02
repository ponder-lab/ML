package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import org.junit.Test;

/**
 * Layers built inside other layers' constructors, in loops and in lazy builds, each class a few
 * {@code super().__init__(...)} levels deep and creating a weight per level, are analysed with a
 * bounded number of nodes (<a href="https://github.com/wala/ML/issues/995">wala/ML#995</a>). The
 * method objects a {@code super()} object exposes are allocated by the super body per calling
 * context, so keying a trampoline's context on such a receiver minted a fresh caller pair per level
 * per construction context, and once those bodies ran with a bound {@code self} everything below
 * the chain multiplied: 394 weight-creation nodes and 785 nodes in all for this program, against 70
 * and 214 when each level dispatches in its caller's context, and 18 and 132 before the bodies ran
 * at all. The bounds sit between the second and the first.
 */
public class TestSuperChainContexts extends TestPythonMLCallGraphShape {

  /** The most nodes the weight-creation summary may have for this program. */
  private static final int MAX_ADD_WEIGHT_NODES = 100;

  /** The most nodes the whole program may have. */
  private static final int MAX_NODES = 300;

  @Test
  public void testFourLevelChainStaysFlat() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_super_chain_contexts.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    int addWeight = 0;
    for (CGNode node : cg)
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/add_weight"))
        addWeight++;
    assertTrue(
        "add_weight nodes: " + addWeight + " > " + MAX_ADD_WEIGHT_NODES,
        addWeight <= MAX_ADD_WEIGHT_NODES);
    assertTrue(
        "nodes: " + cg.getNumberOfNodes() + " > " + MAX_NODES, cg.getNumberOfNodes() <= MAX_NODES);
  }
}
