package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.TRAMPOLINE_METHOD_NAME;
import static java.util.stream.Collectors.joining;

import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.HashMapFactory;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

public abstract class PythonMethodTrampolineTargetSelector<T> implements MethodTargetSelector {

  /**
   * Identifies a trampoline code body by the receiver and the calling layout it was generated for.
   * A body's instructions depend on the call's positional/keyword split and on the keyword names,
   * not just the total argument count; keying on the total collides call sites with equal total
   * arity but different splits (<a href="https://github.com/wala/ML/issues/740">wala/ML#740</a>).
   * Keyword names are a set because binding is by name in both the caller-to-trampoline and
   * trampoline-to-method directions, making bodies insensitive to keyword order.
   *
   * @param receiver The {@link IClass} being dispatched on.
   * @param positionalParameterCount The call's number of positional parameters.
   * @param keywordNames The call's keyword argument names.
   */
  protected record TrampolineKey(
      IClass receiver, int positionalParameterCount, Set<String> keywordNames) {}

  protected final MethodTargetSelector base;

  protected final Map<TrampolineKey, IMethod> codeBodies = HashMapFactory.make();

  public PythonMethodTrampolineTargetSelector(MethodTargetSelector base) {
    this.base = base;
  }

  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    if (receiver != null) {
      Logger logger = this.getLogger();

      logger.fine("Getting callee target for receiver: " + receiver);
      logger.fine("Calling method name is: " + caller.getMethod().getName());

      if (this.shouldProcess(caller, site, receiver)) {
        PythonInvokeInstruction call = this.getCall(caller, site);
        if (call == null) return null;
        TrampolineKey key = this.makeKey(receiver, call);

        if (!codeBodies.containsKey(key)) {
          MethodReference tr =
              MethodReference.findOrCreate(
                  receiver.getReference(),
                  Atom.findOrCreateUnicodeAtom(this.getTrampolineName(call)),
                  AstMethodReference.fnDesc);
          PythonSummary x = new PythonSummary(tr, call.getNumberOfTotalParameters());
          int v = call.getNumberOfTotalParameters() + 1;

          populate(x, v, receiver, call, logger);

          codeBodies.put(key, new PythonSummarizedFunction(tr, x, receiver));
        }

        return codeBodies.get(key);
      }
    }

    return base.getCalleeTarget(caller, site, receiver);
  }

  /**
   * Returns the {@link PythonInvokeInstruction} at the given {@link CallSiteReference} within the
   * given {@link CGNode}.
   *
   * @param caller The calling {@link CGNode}.
   * @param site A {@link CallSiteReference} within the given {@link CGNode}.
   * @return The {@link PythonInvokeInstruction} at the given {@link CallSiteReference} within the
   *     given {@link CGNode}.
   */
  protected PythonInvokeInstruction getCall(CGNode caller, CallSiteReference site) {
    SSAAbstractInvokeInstruction inst = caller.getIR().getCalls(site)[0];
    if (inst instanceof PythonInvokeInstruction) {
      return (PythonInvokeInstruction) inst;
    }
    return null;
  }

  /**
   * Returns the {@link TrampolineKey} identifying the trampoline code body for the given receiver
   * and the given call's positional/keyword layout.
   *
   * @param receiver The {@link IClass} being dispatched on.
   * @param call The {@link PythonInvokeInstruction} whose layout the trampoline is generated for.
   * @return The {@link TrampolineKey} identifying the trampoline code body for the given receiver
   *     and the given call's positional/keyword layout.
   */
  private TrampolineKey makeKey(IClass receiver, PythonInvokeInstruction call) {
    return new TrampolineKey(
        receiver, call.getNumberOfPositionalParameters(), Set.copyOf(call.getKeywords()));
  }

  /**
   * Returns the name of the trampoline method for the given call's positional/keyword layout.
   * Distinct layouts must yield distinct names; otherwise, {@link MethodReference#findOrCreate}
   * canonicalizes distinct code bodies onto one reference (<a
   * href="https://github.com/wala/ML/issues/740">wala/ML#740</a>). Keyword names are sorted so call
   * sites differing only in keyword order share a name, matching {@link TrampolineKey}.
   *
   * @param call The {@link PythonInvokeInstruction} whose layout the trampoline is generated for.
   * @return The name of the trampoline method for the given call's positional/keyword layout.
   */
  private String getTrampolineName(PythonInvokeInstruction call) {
    return TRAMPOLINE_METHOD_NAME
        + call.getNumberOfPositionalParameters()
        + call.getKeywords().stream().sorted().map(k -> "$" + k).collect(joining());
  }

  /**
   * The {@link Logger} to be used.
   *
   * @return The {@link Logger} to be used.
   */
  protected abstract Logger getLogger();

  /**
   * True iff this {@link PythonMethodTrampolineTargetSelector} should handle the given {@link
   * CGNode}, {@link CallSiteReference}, {@link IClass} combination. If the combination is not to be
   * processed, the next target selector will be used.
   *
   * @return True iff this {@link PythonMethodTrampolineTargetSelector} should handle the given
   *     {@link CGNode}, {@link CallSiteReference}, {@link IClass} combination.
   */
  protected abstract boolean shouldProcess(CGNode caller, CallSiteReference site, IClass receiver);

  /**
   * Populate the given {@link PythonSummary} that will be used as the trampoline. At the completion
   * of this method, the given {@link PythonInvokeInstruction} will be the last instruction.
   *
   * <p>This fill the trampoline body that eventually invokes the original method.
   *
   * @param x The {@link PythonSummary} representing the trampoline to fill.
   * @param v The starting variable number in the SSA.
   * @param receiver The receiver of the original call.
   * @param call The original call.
   * @param logger The {@link Logger} to use.
   */
  protected abstract void populate(
      PythonSummary x, int v, IClass receiver, PythonInvokeInstruction call, Logger logger);
}
