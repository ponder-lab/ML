package com.ibm.wala.cast.python.ipa.callgraph;

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
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;
import java.util.logging.Logger;

public abstract class PythonMethodTrampolineTargetSelector<T> implements MethodTargetSelector {

  protected final MethodTargetSelector base;

  protected final Map<Pair<IClass, Integer>, IMethod> codeBodies = HashMapFactory.make();

  public PythonMethodTrampolineTargetSelector(MethodTargetSelector base) {
    super();
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
        Pair<IClass, Integer> key = this.makeKey(receiver, call);

        if (!codeBodies.containsKey(key)) {
          MethodReference tr =
              MethodReference.findOrCreate(
                  receiver.getReference(),
                  Atom.findOrCreateUnicodeAtom("trampoline" + call.getNumberOfTotalParameters()),
                  AstMethodReference.fnDesc);
          PythonSummary x = new PythonSummary(tr, call.getNumberOfTotalParameters());
          int v = call.getNumberOfTotalParameters() + 1;

          populate(x, v, receiver, call, logger);

          PythonSummarizedFunction function = new PythonSummarizedFunction(tr, x, receiver);
          codeBodies.put(key, function);
        }

        return codeBodies.get(key);
      }
    }

    return base.getCalleeTarget(caller, site, receiver);
  }

  protected PythonInvokeInstruction getCall(CGNode caller, CallSiteReference site) {
    return (PythonInvokeInstruction) caller.getIR().getCalls(site)[0];
  }

  private Pair<IClass, Integer> makeKey(IClass receiver, PythonInvokeInstruction call) {
    return Pair.make(receiver, call.getNumberOfTotalParameters());
  }

  protected abstract Logger getLogger();

  protected abstract boolean shouldProcess(CGNode caller, CallSiteReference site, IClass receiver);

  protected abstract void populate(
      PythonSummary x, int v, IClass receiver, PythonInvokeInstruction call, Logger logger);
}
