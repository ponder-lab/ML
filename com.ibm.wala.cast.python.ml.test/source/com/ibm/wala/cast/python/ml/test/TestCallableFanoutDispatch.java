package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import org.junit.Test;

/**
 * Witnesses for wala/ML#869: a callable receiver whose points-to set spans several callable classes
 * dispatches to all of them through the fan-out trampoline, instead of receiving no call edges at
 * all.
 *
 * <p>The fixture's {@code Holder} stores either a parameter-supplied {@code Passed} instance or a
 * same-frame-constructed {@code Direct} instance in one field and calls through it. Under 1-CFA the
 * parameter-supplied holder's field holds BOTH classes (the other {@code __init__} branch is
 * statically feasible), which is the configuration the selector previously answered with silence.
 * {@link #testMultiCandidateReceiverDispatchesToAllCandidates} fails if the fan-out path stops
 * dispatching (the wala/ML#869 defect returning); {@link
 * #testSingletonReceiverStaysOnThePreciseTrampoline} fails if a singleton candidate set starts
 * routing through the fan-out dispatcher, pinning that the precise single-candidate path is
 * untouched. The fan-out trampoline's name carries the candidate-set size ({@code $fanout2}), the
 * graph-visible half of the permanent size diagnostic.
 */
public class TestCallableFanoutDispatch extends TestPythonMLCallGraphShape {

  @Test
  public void testMultiCandidateReceiverDispatchesToAllCandidates() throws Exception {
    CallGraph cg = analyze();
    boolean found = false;

    for (CGNode use : useNodes(cg)) {
      Set<String> reachable = twoHopSuccessorSignatures(cg, use);

      if (reachable.stream().anyMatch(s -> s.contains("$fanout2"))) {
        found = true;
        assertTrue(
            "Expecting the fan-out context to reach the parameter-supplied class's call"
                + " trampoline: "
                + reachable,
            reachable.stream().anyMatch(s -> s.contains("Passed.call.trampoline")));
        assertTrue(
            "Expecting the fan-out context to reach the same-frame class's call trampoline: "
                + reachable,
            reachable.stream().anyMatch(s -> s.contains("Direct.call.trampoline")));
      }
    }

    assertTrue(
        "Expecting a context routed through the fan-out dispatcher, whose name carries the"
            + " candidate-set size as the graph-visible diagnostic.",
        found);
  }

  @Test
  public void testSingletonReceiverStaysOnThePreciseTrampoline() throws Exception {
    CallGraph cg = analyze();
    boolean found = false;

    for (CGNode use : useNodes(cg)) {
      Set<String> successors = new HashSet<>();
      for (Iterator<CGNode> it = cg.getSuccNodes(use); it.hasNext(); )
        successors.add(it.next().getMethod().getSignature());

      if (successors.stream().anyMatch(s -> s.contains("Direct.call.trampoline"))) {
        found = true;
        assertTrue(
            "Expecting the singleton-candidate context to stay on the per-class trampoline, not"
                + " the fan-out dispatcher: "
                + successors,
            successors.stream().noneMatch(s -> s.contains("$fanout")));
        assertTrue(
            "Expecting the singleton-candidate context not to reach the other class: " + successors,
            successors.stream().noneMatch(s -> s.contains("Passed")));
      }
    }

    assertTrue("Expecting a context on the precise single-candidate path.", found);
  }

  private CallGraph analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_layer_field_dispatch.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    return builder.makeCallGraph(builder.getOptions());
  }

  private static Set<CGNode> useNodes(CallGraph cg) {
    Set<CGNode> ret = new HashSet<>();
    for (CGNode node : cg)
      if (node.getMethod().getSignature().contains("Holder.use.do")) ret.add(node);
    return ret;
  }

  private static Set<String> twoHopSuccessorSignatures(CallGraph cg, CGNode node) {
    Set<String> ret = new HashSet<>();
    for (Iterator<CGNode> it = cg.getSuccNodes(node); it.hasNext(); ) {
      CGNode succ = it.next();
      ret.add(succ.getMethod().getSignature());
      for (Iterator<CGNode> it2 = cg.getSuccNodes(succ); it2.hasNext(); )
        ret.add(it2.next().getMethod().getSignature());
    }
    return ret;
  }
}
