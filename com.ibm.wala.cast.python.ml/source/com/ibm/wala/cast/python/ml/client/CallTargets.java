package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * The client-facing authority on which program functions a call runs. A client asks it rather than
 * following the analysis's trampolines and forwarding synthetic nodes itself, which would tie the
 * client to how the analysis lays out its call graph.
 *
 * <p>Program code is a function or method the program defines: an {@link AstMethod} that is not
 * synthetic and is not one of the analysis's library models (a TensorFlow or NumPy summary).
 *
 * @see TensorFlowApis
 */
public final class CallTargets {

  /** The class-name prefixes of the analysis's library models, which are not program code. */
  private static final Set<String> MODEL_CLASS_PREFIXES =
      Set.of("Ltensorflow", "L$tensorflow", "Lnumpy");

  private CallTargets() {}

  /**
   * Returns the program functions a call runs: its call-graph targets that are program code, and
   * the program code it reaches through trampolines, synthetic constructors and library models.
   * That includes callbacks a library model invokes, such as the function {@code tf.map_fn} or
   * {@code dataset.map} applies, and a model's {@code call} that {@code fit} runs. A program
   * function reached is not looked into further: the calls in its body are its own. A call that
   * reaches only library models gives the empty set.
   *
   * <p>This answers which program code may run, not what it receives. A callback receives what the
   * library passes it, not the call's arguments, so this is not for aligning a call's arguments
   * with a function's parameters; {@link #parameterBoundTargets} is.
   *
   * @param builder The builder whose call graph holds the call's targets.
   * @param node The node whose IR holds the call.
   * @param call The call.
   * @return The program functions' nodes.
   */
  public static Set<CGNode> programTargets(
      PropagationCallGraphBuilder builder, CGNode node, SSAAbstractInvokeInstruction call) {
    CallGraph callGraph = builder.getCallGraph();
    Set<CGNode> ret = new LinkedHashSet<>();
    Set<CGNode> seen = new HashSet<>();
    Deque<CGNode> worklist =
        new ArrayDeque<>(callGraph.getPossibleTargets(node, call.getCallSite()));
    while (!worklist.isEmpty()) {
      CGNode target = worklist.pop();
      if (!seen.add(target)) continue;
      if (isProgramCode(target.getMethod())) ret.add(target);
      else
        for (Iterator<CGNode> forwarded = callGraph.getSuccNodes(target); forwarded.hasNext(); )
          worklist.push(forwarded.next());
    }
    return Collections.unmodifiableSet(ret);
  }

  /**
   * Returns the program functions whose parameters receive the call's own arguments, each with
   * whether the analysis binds a receiver before them. A target that is program code receives the
   * arguments as passed. One reached through a trampoline receives what the trampoline forwards:
   * the bound instance or, for a class method, the class, and then the arguments, as {@code
   * obj.m(x)} and {@code C.k(x)} do; or the arguments alone, as a static method called through an
   * instance, {@code obj.s(x)}, does. The flag says which, read from the call the trampoline makes;
   * a trampoline whose forwarding call cannot be read is assumed to bind a receiver, as an instance
   * method's does.
   *
   * <p>This is the subset of {@link #programTargets} a client can align the call's arguments with;
   * a callback a library model invokes receives what the model passes it instead.
   *
   * @param builder The builder whose call graph holds the call's targets.
   * @param node The node whose IR holds the call.
   * @param call The call.
   * @return Each such function's node, mapped to whether a receiver is bound before the call's
   *     arguments.
   */
  public static Map<CGNode, Boolean> parameterBoundTargets(
      PropagationCallGraphBuilder builder, CGNode node, SSAAbstractInvokeInstruction call) {
    CallGraph callGraph = builder.getCallGraph();
    IClassHierarchy cha = builder.getClassHierarchy();
    Map<CGNode, Boolean> ret = new LinkedHashMap<>();
    for (CGNode target : callGraph.getPossibleTargets(node, call.getCallSite())) {
      if (isProgramCode(target.getMethod())) ret.put(target, Boolean.FALSE);
      else if (isTrampoline(cha, target.getMethod()))
        for (Iterator<CGNode> forwarded = callGraph.getSuccNodes(target); forwarded.hasNext(); ) {
          CGNode method = forwarded.next();
          if (isProgramCode(method.getMethod()))
            ret.putIfAbsent(method, bindsReceiver(callGraph, target, method, call));
        }
    }
    return Collections.unmodifiableMap(ret);
  }

  /**
   * Whether a trampoline binds a receiver before the call's arguments when it forwards the call to
   * a function: whether the call it makes passes more positional arguments than the call it
   * receives. A trampoline whose forwarding call cannot be read is taken to bind one, as an
   * instance method's does.
   *
   * @param callGraph The call graph.
   * @param trampoline The trampoline's node.
   * @param method The function's node.
   * @param call The call the trampoline receives.
   * @return {@code true} iff a receiver is bound before the call's arguments.
   */
  private static boolean bindsReceiver(
      CallGraph callGraph, CGNode trampoline, CGNode method, SSAAbstractInvokeInstruction call) {
    if (trampoline.getIR() == null) return true;
    for (SSAInstruction instruction : trampoline.getIR().getInstructions())
      if (instruction instanceof PythonInvokeInstruction forwarding
          && callGraph.getPossibleTargets(trampoline, forwarding.getCallSite()).contains(method)
          && call instanceof PythonInvokeInstruction received)
        return forwarding.getNumberOfPositionalParameters()
            > received.getNumberOfPositionalParameters();
    return true;
  }

  /**
   * Whether a method is a trampoline: a synthetic method that binds its callee's first parameter,
   * the instance of a bound method or the class of a class method, and forwards the call to it.
   *
   * @param cha The class hierarchy.
   * @param method The method.
   * @return {@code true} iff the method is a trampoline.
   */
  static boolean isTrampoline(IClassHierarchy cha, IMethod method) {
    if (method.getSelector().name().toString().startsWith(PythonTypes.TRAMPOLINE_METHOD_NAME))
      return true;
    IClass trampoline = cha.lookupClass(PythonTypes.trampoline);
    return trampoline != null && cha.isSubclassOf(method.getDeclaringClass(), trampoline);
  }

  /**
   * The client-facing authority on whether a call-graph node is program code: a function or method
   * the program defines, which is an {@link AstMethod} that is not synthetic and is not one of the
   * analysis's library models (a TensorFlow or NumPy summary). A client asks this rather than
   * testing a node's class name against the models' packages.
   *
   * @param node The node.
   * @return {@code true} iff the program defines the node's method.
   */
  public static boolean isProgramCode(CGNode node) {
    return isProgramCode(node.getMethod());
  }

  /**
   * Whether a method is program code, as {@link #isProgramCode(CGNode)} decides.
   *
   * @param method The method.
   * @return {@code true} iff the program defines it.
   */
  static boolean isProgramCode(IMethod method) {
    if (!(method instanceof AstMethod) || method.isWalaSynthetic()) return false;
    String name = method.getDeclaringClass().getName().toString();
    for (String prefix : MODEL_CLASS_PREFIXES) if (name.startsWith(prefix)) return false;
    return true;
  }
}
