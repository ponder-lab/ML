package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import org.junit.Test;

/**
 * A base method reached through {@code super()} runs on the one instance whose method called {@code
 * super()} (<a href="https://github.com/wala/ML/issues/1015">wala/ML#1015</a>). Every method body
 * allocates its own {@code super} object carrying that body's instance, so the stub call on it is
 * keyed on the object and the explicit body reads one instance off it; keyed on the caller's
 * context, one stub node collected the super objects of every instance reaching it and the base
 * method's {@code self} was their union, so a base method that calls {@code self.m(...)} back
 * dispatched on every instance and the contexts beneath multiplied by the instance count at each
 * such hop. The fixture has three instances of a subclass whose override calls {@code
 * super().make_features(...)} and whose base {@code make_features} calls {@code
 * self.make_features(...)}.
 */
public class TestSuperSelfBinding extends TestPythonMLCallGraphShape {

  @Test
  public void testBaseMethodBindsOneInstance() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_super_dispatch_recursion.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    int bodies = 0;
    for (CGNode node : cg) {
      String signature = node.getMethod().getSignature();
      if (!signature.contains("TextIn.make_features") || signature.contains("trampoline")) continue;
      bodies++;
      PointerKey self =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(node, node.getIR().getParameter(1));
      assertEquals(
          "Instances bound as `self` in " + node,
          1,
          builder.getPointerAnalysis().getPointsToSet(self).size());
    }
    assertTrue("No base make_features body.", bodies > 0);
  }
}
