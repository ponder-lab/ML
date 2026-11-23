package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * An integer subscript of a tensor allocates its result as a tensor, as a slice of it does, so its
 * value points to an object, and a merge with a Python scalar keeps the tensor (wala/ML#1225).
 */
public class TestTensorIntegerSubscript extends TestPythonMLCallGraphShape {

  private static final String TENSOR = "Ltensorflow/python/framework/ops/Tensor";

  @Test
  public void testIntegerSubscriptsPointToTensors() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_tensor_integer_subscript.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    assertEquals(
        "A row of a tensor is a tensor the subscript allocates.",
        Set.of("program:" + TENSOR),
        parameterKeys(builder, cg, "consume_tensor_index"));
    assertEquals(
        "An element of a tensor's dynamic shape is a tensor the subscript allocates.",
        Set.of("program:" + TENSOR),
        parameterKeys(builder, cg, "consume_shape_index"));
    assertEquals(
        "A row of an int32 tensor, as `edge_index[0]`, is a tensor the subscript allocates.",
        Set.of("program:" + TENSOR),
        parameterKeys(builder, cg, "consume_row_of_index"));
    Set<String> phi = parameterKeys(builder, cg, "consume_phi");
    assertTrue(
        "A merge of a Python scalar and an element of a shape holds the tensor: " + phi,
        phi.contains("program:" + TENSOR));
    assertTrue("The merge holds the scalar too: " + phi, phi.contains("constant:2"));
  }

  private static Set<String> parameterKeys(
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
                : key instanceof ConstantKey<?> constant
                    ? "constant:" + constant.getValue()
                    : "non-allocation:" + key.getClass().getSimpleName());
    }
    return ret;
  }
}
