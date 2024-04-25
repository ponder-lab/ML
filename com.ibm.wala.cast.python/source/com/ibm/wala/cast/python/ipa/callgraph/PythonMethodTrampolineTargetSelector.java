package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;
import java.util.logging.Logger;

public abstract class PythonMethodTrampolineTargetSelector<T> implements MethodTargetSelector {

  protected static void log(Logger logger, CGNode caller, IClass receiver) {
    logger.fine("Getting callee target for receiver: " + receiver);
    logger.fine("Calling method name is: " + caller.getMethod().getName());
  }

  protected final MethodTargetSelector base;

  protected final Map<Pair<IClass, Integer>, IMethod> codeBodies = HashMapFactory.make();

  public PythonMethodTrampolineTargetSelector(MethodTargetSelector base) {
    super();
    this.base = base;
  }

  protected Pair<IClass, Integer> makeKey(IClass receiver, PythonInvokeInstruction call) {
    return Pair.make(receiver, call.getNumberOfTotalParameters());
  }

  protected PythonInvokeInstruction getCall(CGNode caller, CallSiteReference site) {
    return (PythonInvokeInstruction) caller.getIR().getCalls(site)[0];
  }
}
