package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * A loop over a dataset binds an element the iterator's {@code __next__} allocates, and an unpack
 * binds the element's component allocations (<a
 * href="https://github.com/wala/ML/issues/1010">wala/ML#1010</a>), so the values reaching the loop
 * body have points-to sets a generator can read; they used to be the dataset itself, or empty.
 */
public class TestDatasetElementAllocations extends TestPythonMLCallGraphShape {

  @Test
  public void testLoopVariableAndComponentsAreElementAllocations() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_dataset_iteration.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    assertEquals(
        "The whole loop variable is the iterator's element allocation.",
        Set.of("summary:Ltensorflow/data/element"),
        parameterAllocations(builder, cg, "consume_single"));
    assertEquals(
        "The first unpacked component is the element's first component allocation.",
        Set.of("summary:Ltensorflow/data/element_0"),
        parameterAllocations(builder, cg, "consume_x"));
    assertEquals(
        "The second unpacked component is the element's second component allocation.",
        Set.of("summary:Ltensorflow/data/element_1"),
        parameterAllocations(builder, cg, "consume_y"));
  }

  private static Set<String> parameterAllocations(
      PythonSSAPropagationCallGraphBuilder builder, CallGraph cg, String function) {
    Set<String> ret = new HashSet<>();
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().contains("." + function + ".")) continue;
      PointerKey parameter =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      for (InstanceKey key : builder.getPointerAnalysis().getPointsToSet(parameter))
        ret.add(
            key instanceof AllocationSiteInNode allocation
                ? (allocation.getNode().getMethod().isWalaSynthetic() ? "summary:" : "program:")
                    + allocation.getSite().getDeclaredType().getName()
                : "non-allocation:" + key.getClass().getSimpleName());
    }
    return ret;
  }
}
