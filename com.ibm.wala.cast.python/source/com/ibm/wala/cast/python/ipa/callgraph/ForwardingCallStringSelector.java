package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.ContextSelector;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;

/**
 * A call-string selector for program functions that keeps one call site, and two where the caller
 * passes one of its own parameters on (wala/ML#1305).
 *
 * <p>With one site, a helper that a wrapper calls at one site has one node for every caller of the
 * wrapper, so what the helper computes for one caller's argument joins what it computes for
 * another's. The second site separates those callers where the helper's argument comes from the
 * wrapper's own parameters, which is where the caller decides what the helper sees. A call whose
 * arguments are the caller's own values keeps one site, and so does one whose arguments are only
 * computed from its parameters, as in {@code helper(x.data)}.
 *
 * <p>A call made by a trampoline or a summary keeps two sites where the caller's own string has
 * two, so a method reached through a trampoline is separated as a function called directly is.
 */
public class ForwardingCallStringSelector extends CallStringContextSelector {

  /** The prefix of the names of the classes that hold the program's own functions. */
  private static final String SCRIPT_PREFIX = "Lscript ";

  /**
   * Creates the selector.
   *
   * @param base The selector whose contexts the call strings extend.
   */
  public ForwardingCallStringSelector(ContextSelector base) {
    super(base);
  }

  /**
   * Whether the given node is one of the program's own functions, rather than a trampoline or a
   * summary.
   *
   * @param node The node.
   * @return Whether {@code node}'s method is declared in the program.
   */
  public static boolean isProgramFunction(CGNode node) {
    return node.getMethod().getDeclaringClass().getName().toString().startsWith(SCRIPT_PREFIX);
  }

  /**
   * The number of call sites in the given node's call string.
   *
   * @param node The node.
   * @return The length of {@code node}'s call string, or 0 when its context has none.
   */
  public static int callStringLength(CGNode node) {
    CallString cs = (CallString) node.getContext().get(CALL_STRING);
    return cs == null ? 0 : cs.getCallSiteRefs().length;
  }

  /**
   * Whether a call from the given node into the given method recurses: the method is the node's own
   * or one whose call site is on the node's call string.
   *
   * @param caller The calling node.
   * @param target The called method.
   * @return Whether {@code target} is {@code caller}'s method or a method of its call string.
   */
  private static boolean recurses(CGNode caller, IMethod target) {
    if (target.equals(caller.getMethod())) return true;
    CallString cs = (CallString) caller.getContext().get(CALL_STRING);
    if (cs != null) for (IMethod method : cs.getMethods()) if (target.equals(method)) return true;
    return false;
  }

  /**
   * Whether the call at the given site passes one of the caller's own parameters on as an argument.
   *
   * @param caller The calling node.
   * @param site The call site in {@code caller}.
   * @return Whether an argument of the call is a parameter of {@code caller}.
   */
  public static boolean passesParameterOn(CGNode caller, CallSiteReference site) {
    IR ir = caller.getIR();
    if (ir == null) return false;
    // Use 0 is the called function.
    for (SSAAbstractInvokeInstruction call : ir.getCalls(site))
      for (int i = 1; i < call.getNumberOfUses(); i++)
        if (ir.getSymbolTable().isParameter(call.getUse(i))) return true;
    return false;
  }

  @Override
  protected int getLength(CGNode caller, CallSiteReference site, IMethod target) {
    if (!isProgramFunction(caller)) return callStringLength(caller) >= 2 ? 2 : 1;
    // A call into a function already on the caller's string, the caller's own among them, keeps one
    // site: a second would unroll the recursion by one call, whose nodes merge again below it.
    if (recurses(caller, target)) return 1;
    return passesParameterOn(caller, site) ? 2 : 1;
  }
}
