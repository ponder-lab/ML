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
 * Witnesses for wala/ML#869 and wala/ML#1012: a callable receiver whose points-to set spans several
 * callable classes dispatches to every one of them, instead of receiving no call edges at all, and
 * each instance dispatches to its own class's callable directly.
 *
 * <p>The fixture's {@code Holder} stores either a parameter-supplied {@code Passed} instance or a
 * same-frame-constructed {@code Direct} instance in one field and calls through it. Under 1-CFA the
 * parameter-supplied holder's field holds BOTH classes (the other {@code __init__} branch is
 * statically feasible), which is the configuration the selector once answered with silence. The
 * selector then routed such a site through a fan-out trampoline chosen by reading the callee's
 * points-to set mid-solve, which made the target depend on how much of the set had been computed
 * (wala/ML#1012); it now resolves the callable per receiver instance, so both classes' trampolines
 * are direct successors of the call. {@link #testMultiCandidateReceiverDispatchesEachCandidate}
 * fails if either class stops being dispatched or a dispatcher reappears between the call and the
 * classes' trampolines; {@link #testSingletonReceiverStaysOnThePreciseTrampoline} fails if no
 * context reached by one class dispatches that class alone.
 */
public class TestCallableFanoutDispatch extends TestPythonMLCallGraphShape {

  @Test
  public void testMultiCandidateReceiverDispatchesEachCandidate() throws Exception {
    CallGraph cg = analyze();
    boolean found = false;

    for (CGNode use : useNodes(cg)) {
      Set<String> successors = successorSignatures(cg, use);

      if (successors.stream().anyMatch(s -> s.contains("Passed.call.trampoline"))
          && successors.stream().anyMatch(s -> s.contains("Direct.call.trampoline"))) {
        found = true;
        assertTrue(
            "Expecting each candidate to dispatch to its own class's trampoline directly: "
                + successors,
            successors.stream().noneMatch(s -> s.contains("$fanout")));
      }
    }

    assertTrue(
        "Expecting a context that dispatches both the parameter-supplied class's and the"
            + " same-frame class's call trampolines directly.",
        found);
  }

  /**
   * The holder constructed in the module's frame ({@code h1}) holds a {@code Direct} instance only,
   * so its context dispatches {@code Direct}'s trampoline alone: no other class and no dispatcher
   * between the call and the trampoline.
   */
  @Test
  public void testSingletonReceiverStaysOnThePreciseTrampoline() throws Exception {
    CallGraph cg = analyze();
    boolean found = false;

    for (CGNode use : useNodes(cg)) {
      Set<String> successors = successorSignatures(cg, use);
      if (successors.stream().anyMatch(s -> s.contains("Direct.call.trampoline"))
          && successors.stream().noneMatch(s -> s.contains("Passed"))
          && successors.stream().noneMatch(s -> s.contains("$fanout"))) found = true;
    }

    assertTrue(
        "Expecting a context that dispatches the same-frame class's trampoline alone.", found);
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

  private static Set<String> successorSignatures(CallGraph cg, CGNode node) {
    Set<String> ret = new HashSet<>();
    for (Iterator<CGNode> it = cg.getSuccNodes(node); it.hasNext(); )
      ret.add(it.next().getMethod().getSignature());
    return ret;
  }
}
