package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * A builder's per-builder memos don't outlive its analysis (<a
 * href="https://github.com/wala/ML/issues/1170">wala/ML#1170</a>). Their values reach the builder,
 * so an entry left behind keeps the builder, its call graph and its class hierarchy alive.
 */
public class BuilderReleaseTest extends TestPythonMLCallGraphShape {

  /**
   * Analyzes a fixture.
   *
   * @param file The fixture.
   * @return The builder the analysis ran on.
   * @throws Exception On analysis failure.
   */
  private PythonSSAPropagationCallGraphBuilder analyze(String file) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    return builder;
  }

  /**
   * Queries the {@code consume_pair} parameter's dataset component after the analysis, as a client
   * reads a {@code fit} step's {@code inputs}.
   *
   * @param builder The builder the analysis ran on.
   */
  private static void queryComponent(PythonSSAPropagationCallGraphBuilder builder) {
    List<PointerKey> parameters = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/consume_pair"))
        parameters.add(builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for consume_pair.", 1, parameters.size());

    ContainerElements elements = ContainerElements.ofValue(builder, parameters.get(0));
    assertTrue("The query reads the component as a declared pair.", elements.arity().isPresent());
  }

  /**
   * A dataset component queried after the analysis memoizes its generator dispatches, as the
   * analysis does, so a client querying many values doesn't dispatch each afresh (wala/ML#1176).
   * The builder holds the memo.
   */
  @Test
  public void testPostAnalysisQueryMemoizesDispatches() throws Exception {
    PythonSSAPropagationCallGraphBuilder builder =
        this.analyze("tf2_test_container_elements_of_value.py");
    try {
      assertFalse(
          "The analysis clears its dispatches.", TensorGeneratorFactory.hasDispatches(builder));
      queryComponent(builder);
      assertTrue(
          "A query made after the analysis memoizes its dispatches.",
          TensorGeneratorFactory.hasDispatches(builder));
    } finally {
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * A builder queried after the analysis is collected once its client drops it, without the client
   * clearing any memo: the memoized dispatches don't keep it alive (wala/ML#1170).
   */
  @Test
  public void testQueriedBuilderIsCollected() throws Exception {
    PythonSSAPropagationCallGraphBuilder builder =
        this.analyze("tf2_test_container_elements_of_value.py");
    queryComponent(builder);
    assertTrue("The query memoized dispatches.", TensorGeneratorFactory.hasDispatches(builder));
    WeakReference<PythonSSAPropagationCallGraphBuilder> reference = new WeakReference<>(builder);
    builder = null;
    for (int i = 0; i < 50 && reference.get() != null; i++) {
      System.gc();
      Thread.sleep(20);
    }
    assertNull("The builder is collected.", reference.get());
  }

  /**
   * The memoized class axes of {@code flow_from_directory} labels are released with the other
   * per-builder memos when the analysis ends.
   */
  @Test
  public void testAnalysisReleasesClassAxes() throws Exception {
    PythonSSAPropagationCallGraphBuilder builder =
        this.analyze("tf2_test_categorical_labels_from_loss.py");
    assertFalse(
        "The analysis clears its memoized class axes.",
        FlowFromDirectoryGenerator.hasClassAxes(builder));
  }
}
