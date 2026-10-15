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
 * What a Keras weight created by {@code add_weight} is allocated as.
 *
 * <p>In TensorFlow, {@code add_weight} returns a {@code tf.Variable}, and a client tells a variable
 * from a tensor by its allocation, so this asserts on the allocation rather than on a type: the
 * weight's tensor type is the same either way.
 */
public class TestAddWeightAllocation extends TestPythonMLCallGraphShape {

  /**
   * The weight reaching {@code weight_sink} is allocated as a tensor by the {@code add_weight}
   * summary, not as a variable.
   *
   * <p>TODO: Expect {@code Ltensorflow/python/ops/variables/Variable} once <a
   * href="https://github.com/wala/ML/issues/1068">wala/ML#1068</a> is fixed.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAddWeightAllocatesATensor() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_add_weight_variable.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    assertEquals(
        Set.of("summary:Ltensorflow/python/framework/ops/Tensor"),
        parameterAllocations(builder, cg, "weight_sink"));
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
