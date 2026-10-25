package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.io.File;
import java.util.Collections;
import java.util.Set;
import java.util.TreeSet;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests that the calls of the gradient APIs that work only while a graph is being built name their
 * APIs ({@link TensorFlowApis}), on {@code tf2_test_graph_gradients.py} (<a
 * href="https://github.com/wala/ML/issues/1144">wala/ML#1144</a>). An optimizer's methods are named
 * by the optimizer's model, so a call of one reached through an instance names it.
 */
public class TestGraphGradientApis extends TestPythonMLCallGraphShape {

  private static PythonSSAPropagationCallGraphBuilder builder;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        new TestGraphGradientApis()
            .makeEngine(Collections.<File>emptyList(), "tf2_test_graph_gradients.py");
    builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
  }

  /**
   * The APIs the calls of a fixture function name, across every node of the function.
   *
   * @param function The function's name.
   * @return The APIs' names.
   */
  private static Set<String> apisIn(String function) {
    Set<String> apis = new TreeSet<>();
    boolean found = false;
    for (CGNode node : builder.getCallGraph()) {
      if (!CallTargets.isProgramCode(node)
          || !node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + function)
          || node.getIR() == null) continue;
      found = true;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof SSAAbstractInvokeInstruction call)
          apis.addAll(TensorFlowApis.of(builder, node, call));
    }
    assertFalse("`" + function + "` must be in the call graph.", !found);
    return apis;
  }

  @Test
  public void testGetGradients() {
    assertTrue(
        apisIn("step").toString(),
        apisIn("step").contains("tensorflow.keras.optimizers.Optimizer.get_gradients"));
  }

  @Test
  public void testGetUpdates() {
    assertTrue(apisIn("step").contains("tensorflow.keras.optimizers.Optimizer.get_updates"));
  }

  @Test
  public void testGradients() {
    assertTrue(apisIn("step").contains("tensorflow.gradients"));
  }

  @Test
  public void testBackendGradients() {
    assertTrue(apisIn("step").contains("tensorflow.keras.backend.gradients"));
  }

  /**
   * {@code tf.gradients} read through an alias, {@code from tensorflow import gradients}, names the
   * API by the model it reaches, not by the call's spelling.
   */
  @Test
  public void testAliasedGradients() {
    Set<String> apis = apisIn("aliased_step");
    assertTrue(apis.toString(), apis.contains("tensorflow.gradients"));
  }
}
