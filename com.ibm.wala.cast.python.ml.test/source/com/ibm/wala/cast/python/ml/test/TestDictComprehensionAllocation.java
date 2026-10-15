package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;

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
 * The allocation a dict comprehension makes.
 *
 * <p>A dict a comprehension builds is passed to the same function as a dict literal, so the
 * function's parameter shows which allocations the pointer analysis sees.
 */
public class TestDictComprehensionAllocation extends TestPythonMLCallGraphShape {

  /**
   * The parameter points to the literal's {@code dict} allocation only: the comprehension allocates
   * a class whose name does not resolve, so it has no instance key and its dict never reaches the
   * parameter.
   *
   * <p>TODO: Expect the comprehension's {@code dict} allocation too once <a
   * href="https://github.com/wala/ML/issues/1075">wala/ML#1075</a> is fixed.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testComprehensionDictHasNoAllocation() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_dict_comprehension_allocation.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    assertEquals(Set.of("literal:Ldict"), parameterAllocations(builder, cg, "consume"));
  }

  /**
   * The allocations the first parameter of a named function may point to, unioned over its
   * context-sensitive nodes, each named by the allocating function and the allocated type.
   *
   * @param builder The call graph builder.
   * @param cg The call graph.
   * @param function The function's name.
   * @return One entry per allocating function and type; a member that is not an allocation is
   *     recorded.
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
                ? allocation.getNode().getMethod().getDeclaringClass().getName().getClassName()
                    + ":"
                    + allocation.getSite().getDeclaredType().getName()
                : "non-allocation:" + key.getClass().getSimpleName());
    }
    return ret;
  }
}
