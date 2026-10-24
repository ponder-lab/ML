package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import java.io.File;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests that a call to a built-in Keras layer reaches the layer's variable creation, as its first
 * call does at run time (<a href="https://github.com/wala/ML/issues/1145">wala/ML#1145</a>), on
 * {@code tf2_test_builtin_layer_build.py}. Each model's {@code call} makes its sublayer's first
 * call, which TensorFlow 2.9.3 confirms builds the sublayer.
 */
public class TestBuiltinLayerBuild extends TestPythonMLCallGraphShape {

  /** The name of the model of {@code Layer.add_weight}, which creates a layer's variables. */
  private static final String ADD_WEIGHT = "Ltensorflow/keras/layers/Layer/add_weight";

  private static CallGraph callGraph;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        new TestBuiltinLayerBuild()
            .makeEngine(Collections.<File>emptyList(), "tf2_test_builtin_layer_build.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    callGraph = builder.makeCallGraph(builder.getOptions());
  }

  /**
   * Whether a model's {@code call} reaches {@code add_weight} in the call graph.
   *
   * @param model The model class's name.
   * @return {@code true} iff some node of the model's {@code call} reaches an {@code add_weight}
   *     node.
   */
  private static boolean reachesAddWeight(String model) {
    Deque<CGNode> worklist = new ArrayDeque<>();
    for (CGNode node : callGraph)
      if (CallTargets.isProgramCode(node)
          && node.getMethod()
              .getDeclaringClass()
              .getName()
              .toString()
              .endsWith("/" + model + "/call")) worklist.add(node);
    assertFalse("`" + model + ".call` must be in the call graph.", worklist.isEmpty());
    Set<CGNode> seen = new HashSet<>();
    while (!worklist.isEmpty()) {
      CGNode node = worklist.pop();
      if (!seen.add(node)) continue;
      if (node.getMethod().getDeclaringClass().getName().toString().equals(ADD_WEIGHT)) return true;
      for (Iterator<CGNode> successors = callGraph.getSuccNodes(node); successors.hasNext(); )
        worklist.push(successors.next());
    }
    return false;
  }

  /** A {@code Dense} sublayer's first call creates its kernel and bias. */
  @Test
  public void testDense() {
    assertTrue(reachesAddWeight("DenseModel"));
  }

  /**
   * An {@code RNN} over an {@code LSTMCell} creates the cell's variables when it is first called.
   */
  @Test
  public void testRecurrent() {
    assertTrue(reachesAddWeight("RecurrentModel"));
  }

  /** A {@code Flatten} sublayer has no variables, and its call creates none. */
  @Test
  public void testFlatten() {
    assertFalse(reachesAddWeight("FlattenModel"));
  }
}
