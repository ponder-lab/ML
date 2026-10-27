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
 * Tests that a built-in Keras layer builds on {@code __call__} and not on {@code call} (<a
 * href="https://github.com/wala/ML/issues/1166">wala/ML#1166</a>), on {@code
 * tf2_test_builtin_layer_subclass_build.py}. Keras builds a layer in {@code Layer.__call__} before
 * it calls {@code call}, so a subclass's {@code call} creates no variable, whether or not it runs
 * {@code super().call(...)}, while calling the layer does, as TensorFlow 2.9.3 confirms.
 */
public class TestBuiltinLayerSubclassBuild extends TestPythonMLCallGraphShape {

  /** The name of the model of {@code Layer.add_weight}, which creates a layer's variables. */
  private static final String ADD_WEIGHT = "Ltensorflow/keras/layers/Layer/add_weight";

  private static PythonSSAPropagationCallGraphBuilder builder;

  private static CallGraph callGraph;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        new TestBuiltinLayerSubclassBuild()
            .makeEngine(Collections.<File>emptyList(), "tf2_test_builtin_layer_subclass_build.py");
    builder = engine.defaultCallGraphBuilder();
    callGraph = builder.makeCallGraph(builder.getOptions());
  }

  /**
   * Whether a program class's {@code call} reaches {@code add_weight} in the call graph.
   *
   * @param type The program class's name.
   * @return {@code true} iff some node of the class's {@code call} reaches an {@code add_weight}
   *     node.
   */
  private static boolean reachesAddWeight(String type) {
    Deque<CGNode> worklist = new ArrayDeque<>();
    for (CGNode node : callGraph)
      if (CallTargets.isProgramCode(node)
          && node.getMethod()
              .getDeclaringClass()
              .getName()
              .toString()
              .endsWith("/" + type + "/call")) worklist.add(node);
    assertFalse("`" + type + ".call` must be in the call graph.", worklist.isEmpty());
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

  /**
   * A subclass's {@code call} running {@code super().call(...)} creates no variable: the built-in
   * layer's {@code call} doesn't build.
   */
  @Test
  public void testSubclassCallBuildsNothing() {
    assertFalse(reachesAddWeight("BatchNormalization"));
  }

  /** Calling the subclass's instance builds it, as {@code Layer.__call__} does. */
  @Test
  public void testSubclassLayerCallBuilds() {
    assertTrue(reachesAddWeight("NormalizedModel"));
  }

  /**
   * A subclass's {@code call} that skips {@code super().call(...)}, reading the weight the layer
   * keeps, creates no variable either.
   */
  @Test
  public void testSubclassCallWithoutSuperBuildsNothing() {
    assertFalse(reachesAddWeight("TFEmbedding"));
  }

  /**
   * Calling an instance of a subclass whose {@code call} skips {@code super().call(...)} builds it
   * too, since {@code Layer.__call__} builds before it calls {@code call}.
   */
  @Test
  public void testSubclassWithoutSuperLayerCallBuilds() {
    assertTrue(reachesAddWeight("Embeddings"));
  }

  /** Control: calling a built-in {@code Dense} builds it. */
  @Test
  public void testDenseLayerCallBuilds() {
    assertTrue(reachesAddWeight("DenseModel"));
  }
}
