package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;

public abstract class PythonMethodTrampolineTargetSelector<T> implements MethodTargetSelector {

  protected final MethodTargetSelector base;

  protected final Map<Pair<IClass, Integer>, IMethod> codeBodies = HashMapFactory.make();

  public PythonMethodTrampolineTargetSelector(MethodTargetSelector base) {
    super();
    this.base = base;
  }
}
