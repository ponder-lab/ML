package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * A read after the analysis skips the branch-fold work whose result it discards, and a dataset
 * component's elements are read with one dispatch (<a
 * href="https://github.com/wala/ML/issues/1186">wala/ML#1186</a>). Neither changes an answer.
 */
public class AfterAnalysisReadShortcutsTest extends TestPythonMLCallGraphShape {

  /**
   * Fixtures whose reads after the analysis dispatch through callee returns, φs and branch folds,
   * and read dataset components. {@code tf2_test_mode_guard.py} prunes a callee's returns by its
   * call sites' constants (wala/ML#746), and {@code tf2_test_rank_guard_chain_short.py} prunes a
   * call site by a guard before it and has a guard on its result after it.
   */
  private static final List<String> FIXTURES =
      List.of(
          "tf2_test_mode_guard.py",
          "tf2_test_container_elements_of_value.py",
          "tf2_test_fit_structured_component.py",
          "tf2_test_encoder_training_path.py",
          "tf2_test_subscript_param.py",
          "tf2_test_branch_comparisons.py",
          "tf2_test_rank_guard_chain_short.py");

  /**
   * The nodes of the program's own code, rather than of a summary.
   *
   * @param builder The builder.
   * @return The nodes whose IR the script defines.
   */
  private static List<CGNode> scriptNodes(PythonSSAPropagationCallGraphBuilder builder) {
    List<CGNode> ret = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getIR() != null
          && node.getMethod().getDeclaringClass().getName().toString().startsWith("Lscript "))
        ret.add(node);
    return ret;
  }

  /**
   * Renders a container's elements.
   *
   * @param elements The elements, or {@code null}.
   * @param analysis The analysis whose types the positions read.
   * @return The rendering.
   */
  private static String render(ContainerElements elements, TensorTypeAnalysis analysis) {
    if (elements == null) return "none";
    return elements.arity()
        + " "
        + elements.isComplete()
        + " "
        + elements.positionTypes(analysis)
        + " "
        + elements.elementTypes(analysis);
  }

  /**
   * Reads, after the analysis, every value of the script's code: its generator's types, its
   * elements, and its elements as a dataset component.
   *
   * @param builder The builder.
   * @param analysis The analysis.
   * @return Each value's answers, in a fixed order.
   */
  private static Map<String, String> readAll(
      PythonSSAPropagationCallGraphBuilder builder, TensorTypeAnalysis analysis) {
    Map<String, String> ret = new LinkedHashMap<>();
    for (CGNode node : scriptNodes(builder)) {
      IR ir = node.getIR();
      for (int vn = 1; vn <= ir.getSymbolTable().getMaxValueNumber(); vn++) {
        PointerKey value =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
        if (builder.getPropagationSystem().isImplicit(value)) continue;
        String types;
        try {
          TensorGenerator generator =
              TensorGeneratorFactory.getGenerator(
                  builder.getPropagationSystem().findOrCreatePointsToSet(value), builder);
          types = generator == null ? "none" : String.valueOf(generator.getTensorTypes(builder));
        } catch (RuntimeException e) {
          types = e.getClass().getName();
        }
        ContainerElements elements = ContainerElements.ofValue(builder, value);
        ContainerElements component = ContainerElements.ofDatasetComponent(builder, value);
        if (ContainerElements.datasetComponent(builder, value) == null)
          assertNull("A value that is no dataset component has no component elements.", component);
        else
          assertEquals(
              "A dataset component's elements read the same either way.",
              render(elements, analysis),
              render(component, analysis));
        ret.put(
            node + " v" + vn,
            types + " | " + render(elements, analysis) + " | " + render(component, analysis));
      }
    }
    return ret;
  }

  /**
   * The answers read after the analysis are the same with the shortcuts as without them, and the
   * shortcuts skip work on these fixtures.
   */
  @Test
  public void testShortcutsLeaveEveryAnswer() throws Exception {
    long skipped = 0;
    for (String fixture : FIXTURES) {
      PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), fixture);
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      builder.makeCallGraph(builder.getOptions());
      TensorTypeAnalysis analysis = engine.performAnalysis(builder);
      try {
        Map<String, String> without;
        TensorGenerator.afterAnalysisShortcuts = false;
        try {
          without = readAll(builder, analysis);
        } finally {
          TensorGenerator.afterAnalysisShortcuts = true;
        }
        long before = TensorGenerator.AFTER_ANALYSIS_SKIPS.get();
        Map<String, String> with = readAll(builder, analysis);
        skipped += TensorGenerator.AFTER_ANALYSIS_SKIPS.get() - before;
        assertEquals("Answers read after analyzing " + fixture + ".", without, with);
      } finally {
        TensorGenerator.clearCaches(builder);
      }
    }
    assertTrue("Expecting the shortcuts to skip work on the fixtures.", skipped > 0);
  }
}
