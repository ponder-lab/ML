package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * The methods {@code super()} exposes in a class with two program-defined bases dispatch along
 * Python's method resolution order (<a
 * href="https://github.com/wala/ML/issues/1014">wala/ML#1014</a>). The fixture's {@code
 * ExampleInputter(ParallelInputter, ExampleInputterAdapter)} reaches each of {@code __init__},
 * {@code initialize}, {@code make_dataset}, {@code get_dataset_size} and {@code make_features} only
 * through {@code super()}, so a run that drops those dispatches is missing the base methods' nodes
 * altogether. Such a run was once observed on a whole project, varying between runs with the same
 * input; it has not been reproduced since, and this pins the correct outcome.
 */
public class TestSuperMultibaseDispatch extends TestPythonMLCallGraphShape {

  /** The base methods reached only through {@code ExampleInputter}'s {@code super()}. */
  private static final List<String> SUPER_TARGETS =
      List.of(
          "ParallelInputter/__init__",
          "MultiInputter/initialize",
          "ParallelInputter/make_dataset",
          "ParallelInputter/get_dataset_size",
          "ParallelInputter/make_features");

  @Test
  public void testSuperDispatchesAlongResolutionOrder() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_super_multibase_dispatch.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    Set<String> reached = new HashSet<>();
    for (CGNode node : cg) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      for (String target : SUPER_TARGETS) if (name.endsWith(target)) reached.add(target);
    }
    for (String target : SUPER_TARGETS)
      assertTrue(target + " is reached through super().", reached.contains(target));
  }
}
