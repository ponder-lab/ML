package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * A helper that recurses through its own comprehensions keeps one call site for each recursive
 * call, as a helper calling itself does (wala/ML#1305). A comprehension runs in its enclosing
 * function's context, so with a second site its call into the function named the wrapper that
 * reached the function, and every wrapper minted a node of the function and of each comprehension
 * below it. Those nodes form a cycle the generator dispatch walks once per path, which kept a
 * whole-project analysis of a program with such configuration helpers from finishing.
 */
public class TestNestedRecursionContexts extends TestPythonMLCallGraphShape {

  /** The name of the class holding the helper's code. */
  private static final String HELPER = "Lscript tf2_test_nested_recursion_contexts.py/map_values";

  @Test
  public void testRecursionThroughComprehensionKeepsOneSite() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_nested_recursion_contexts.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);
    Set<String> wrappers = new HashSet<>();
    int recursive = 0;
    for (CGNode node : cg) {
      if (!node.getMethod().getDeclaringClass().getName().toString().equals(HELPER)) continue;
      CallString cs = (CallString) node.getContext().get(CallStringContextSelector.CALL_STRING);
      assertNotNull("No call string for " + node, cs);
      CallSiteReference[] sites = cs.getCallSiteRefs();
      IMethod[] methods = cs.getMethods();
      String caller = methods[0].getDeclaringClass().getName().toString();
      if (caller.startsWith(HELPER + "/")) {
        recursive++;
        assertEquals("Sites of a recursive call: " + node, 1, sites.length);
      } else wrappers.add(caller);
    }
    assertTrue("No recursive call reached the helper", recursive > 0);
    // The wrappers pass their parameter on, so each still reaches the helper as a node of its own.
    assertTrue("Wrappers reaching the helper: " + wrappers, wrappers.size() >= 2);
  }
}
