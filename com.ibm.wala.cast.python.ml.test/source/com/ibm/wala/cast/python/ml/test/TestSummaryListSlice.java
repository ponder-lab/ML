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
 * A constant slice of a list a summary returns (<a
 * href="https://github.com/wala/ML/issues/993">wala/ML#993</a>).
 *
 * <p>A summary's list stands in for a value whose length it does not model, so slicing it must not
 * mint a known-length collection in the program. A consumer telling a list the program builds from
 * one a TensorFlow operation returns reads the ALLOCATION, so this asserts on the allocating method
 * rather than on a type: the tensor type of the parameter is the same either way.
 */
public class TestSummaryListSlice extends TestPythonMLCallGraphShape {

  /**
   * The parameter receiving {@code tf.unstack(...)[0:4]} holds only the summary's list, not a
   * collection allocated in the program.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSliceOfSummaryListKeepsTheSummaryList() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_unstack_slice.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);

    assertEquals(
        "The slice of a summary's list is that list, allocated by the summary.",
        Set.of("summary:Llist"),
        parameterAllocations(builder, cg, "four"));
  }

  /**
   * Where the first parameter of a named function may have been allocated, unioned over its
   * context-sensitive nodes: by a summary (a synthetic method) or by the program.
   *
   * @param builder The call graph builder.
   * @param cg The call graph.
   * @param function The function's name.
   * @return One entry per allocating side and type; a member that is not an allocation is recorded.
   */
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
