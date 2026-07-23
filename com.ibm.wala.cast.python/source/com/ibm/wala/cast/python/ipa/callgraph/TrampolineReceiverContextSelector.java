/*
 * Copyright (c) 2018 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.ContextKey;
import com.ibm.wala.ipa.callgraph.ContextSelector;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.ReceiverInstanceContext;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallerSiteContext;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallerSiteContextPair;
import com.ibm.wala.util.intset.IntSet;
import java.util.logging.Logger;

/**
 * Keeps per-receiver state separate through Python's method-trampoline dispatch (<a
 * href="https://github.com/wala/ML/issues/679">wala/ML#679</a>).
 *
 * <p>All instances of a class share one trampoline method per arity, so a call-string context
 * collapses every receiver into a single trampoline node whose {@code $self} unions the instances —
 * and every method body reached through it (e.g., the Keras lazy {@code build}) unions per-instance
 * state across receivers. Call strings cannot repair this at any depth because they key on the
 * caller's <em>method</em>, not its node.
 *
 * <p>The selector applies four rules, in order, before delegating to the base selector:
 *
 * <ol>
 *   <li>calls made from a synthesized constructor ({@link PythonConstructorFunction}) inherit the
 *       constructor's context, keeping per-construction-site argument values separate (<a
 *       href="https://github.com/wala/ML/issues/671">wala/ML#671</a>);
 *   <li>a trampoline callee ({@link PythonInstanceMethodTrampoline}) is keyed on the dispatched
 *       receiver instance, paired with the calling node and site so distinct call sites of one
 *       instance also stay separate (<a
 *       href="https://github.com/wala/ML/issues/530">wala/ML#530</a>);
 *   <li>the real method body dispatched from a per-receiver trampoline node inherits the
 *       trampoline's context, since a call string would re-collapse it;
 *   <li>any other call made from a receiver-keyed node, or from a node whose own context is a
 *       caller-site pair, is keyed on the calling node and site, so per-caller-node separation
 *       propagates down the helper chains under a receiver context (wala/ML#742) — in particular
 *       the constructors (synthesized or XML-summary) allocating sublayers in the Keras lazy {@code
 *       build}, the summary methods those sublayers dispatch, and the module-level helper functions
 *       sibling layer methods share.
 * </ol>
 *
 * <p>Termination: receiver contexts do not chain unboundedly. Dispatching on a receiver already
 * recorded in the caller's context reuses the caller's context, and {@link #MAX_RECEIVER_DEPTH}
 * caps the caller-pair chain in both rule 2 and rule 4; the cap is load-bearing for rule 4's
 * propagation, which would otherwise grow one pair per call through recursive helpers.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TrampolineReceiverContextSelector implements ContextSelector {

  private static final Logger LOGGER =
      Logger.getLogger(TrampolineReceiverContextSelector.class.getName());

  /**
   * Maximum depth of nested caller-pair contexts before receiver keying degrades to context
   * inheritance. A backstop against unbounded context towers under mutually recursive dispatch
   * across distinct instances; ordinary layer nesting stays far below it.
   */
  private static final int MAX_RECEIVER_DEPTH = 8;

  /** The selector handling every other call. */
  private final ContextSelector base;

  /**
   * Constructs a {@link TrampolineReceiverContextSelector}.
   *
   * @param base The selector handling every other call.
   */
  public TrampolineReceiverContextSelector(ContextSelector base) {
    this.base = base;
  }

  @Override
  public Context getCalleeTarget(
      CGNode caller, CallSiteReference site, IMethod callee, InstanceKey[] actualParameters) {
    // Calls made from a synthesized constructor inherit its context (wala/ML#671).
    if (caller.getMethod() instanceof PythonConstructorFunction) return caller.getContext();

    // A trampoline callee is keyed on the dispatched receiver instance (wala/ML#679).
    if (callee.getDeclaringClass() instanceof PythonInstanceMethodTrampoline
        && actualParameters != null
        && actualParameters.length > 0
        && actualParameters[0] != null) {
      InstanceKey receiver = actualParameters[0];

      // Recursive dispatch on the caller's own receiver: reuse the caller's context so
      // self-recursive methods do not grow the context.
      if (receiver.equals(caller.getContext().get(ContextKey.RECEIVER))) return caller.getContext();

      if (receiverDepth(caller) >= MAX_RECEIVER_DEPTH) {
        LOGGER.fine(
            () ->
                "Receiver-context depth cap reached at caller: "
                    + caller
                    + "; inheriting instead of keying on receiver: "
                    + receiver
                    + ".");
        return caller.getContext();
      }

      LOGGER.fine(() -> "Keying trampoline: " + callee + " on receiver: " + receiver + ".");
      return new CallerSiteContextPair(caller, site, new ReceiverInstanceContext(receiver));
    }

    // The real method body dispatched from a per-receiver trampoline node stays per-receiver.
    if (caller.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline)
      return caller.getContext();

    // Calls made from a receiver-keyed method body, or from any node whose own context is a
    // caller-site pair, are keyed on the calling node and site, so per-caller-node separation
    // propagates down the helper chains below a receiver context instead of stopping one hop
    // beneath it (wala/ML#742): a shared module-level helper called by sibling layer methods
    // otherwise collapses onto method-keyed call strings, and every read through it unions the
    // callers' operands (the vendored einsum-via-matmul shim was the witness, minting mixed-rank
    // matmul results from the unioned operands). The keying is the calling node and site alone:
    // pairing in the base selector's call-string context multiplied every keyed node by its
    // base-context tail for ~36% of the layer-heavy whole-project wall clock, with no witness in
    // the suite distinguishing the pair from the bare caller-site key (wala/ML#689). {@link
    // #MAX_RECEIVER_DEPTH} bounds the chain, degrading to the base selector past it; the cap is
    // load-bearing here, since recursive helpers would otherwise grow one pair per recursive
    // call.
    if (caller.getContext().get(ContextKey.RECEIVER) != null
        || caller.getContext() instanceof CallerSiteContext) {
      // The guard is recursion, not depth: a deep-but-acyclic layer stack (a BERT encoder tower)
      // legitimately chains many pairs, while a callee already on the caller chain would grow one
      // pair per recursive call. Degrade to the base selector on a method repeat, with an
      // absolute backstop against pathological acyclic towers.
      if (chainsThroughMethod(caller, callee)) {
        LOGGER.fine(
            () ->
                "Caller-pair recursion (or backstop) at caller: "
                    + caller
                    + "; delegating to the base selector for callee: "
                    + callee
                    + ".");
        return base.getCalleeTarget(caller, site, callee, actualParameters);
      }
      return new CallerSiteContext(caller, site);
    }

    return base.getCalleeTarget(caller, site, callee, actualParameters);
  }

  @Override
  public IntSet getRelevantParameters(CGNode caller, CallSiteReference site) {
    return base.getRelevantParameters(caller, site);
  }

  /**
   * Returns the depth of the caller-pair context chain rooted at the given node.
   *
   * @param node The node whose context chain to measure.
   * @return The number of {@link CallerSiteContext} links reachable by walking callers from the
   *     given node's context, up to {@link #MAX_RECEIVER_DEPTH}.
   */
  private static int receiverDepth(CGNode node) {
    int depth = 0;
    Context c = node.getContext();
    while (depth < MAX_RECEIVER_DEPTH && c instanceof CallerSiteContext) {
      CGNode caller = ((CallerSiteContext) c).getCaller();
      c = caller.getContext();
      depth++;
    }
    return depth;
  }

  /**
   * The absolute bound on rule 4's caller-pair chain length: a backstop against pathological
   * acyclic call towers, far above any real layer stack's nesting.
   */
  private static final int MAX_CALLER_PAIR_DEPTH = 64;

  /**
   * Returns whether keying the given callee on the given caller would recurse: the callee's method
   * already appears on the caller-pair chain (so each recursive call would add a pair), or the
   * chain has reached {@link #MAX_CALLER_PAIR_DEPTH}.
   *
   * @param caller The calling {@link CGNode}.
   * @param callee The dispatched callee.
   * @return {@code true} iff pairing would chain through a repeated method or exceed the backstop.
   */
  private static boolean chainsThroughMethod(CGNode caller, IMethod callee) {
    if (caller.getMethod().equals(callee)) return true; // Direct self-recursion.
    int depth = 0;
    Context c = caller.getContext();
    while (c instanceof CallerSiteContext) {
      if (++depth >= MAX_CALLER_PAIR_DEPTH) return true; // Fail closed at the backstop.
      CGNode up = ((CallerSiteContext) c).getCaller();
      if (up.getMethod().equals(callee)) return true; // The callee is already on the chain.
      c = up.getContext();
    }
    return false;
  }
}
