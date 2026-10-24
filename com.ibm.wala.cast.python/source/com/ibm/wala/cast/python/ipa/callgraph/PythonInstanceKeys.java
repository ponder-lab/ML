package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.SSAContextInterpreter;
import com.ibm.wala.ipa.callgraph.propagation.cfa.ZeroXInstanceKeys;
import com.ibm.wala.ipa.cha.IClassHierarchy;

/**
 * Python's instance keys: {@link ZeroXInstanceKeys}, except that an allocation of a Python scalar
 * type ({@link PythonTypes#isScalar(com.ibm.wala.types.TypeReference)}) is one {@link
 * ConcreteTypeKey} per type, whatever its site or context. A scalar of unknown value, such as the
 * result of {@code len(xs)}, thus points to "some {@code int}", as WALA's JavaScript front end
 * gives each primitive type one abstract object, and literals keep their constant keys (<a
 * href="https://github.com/wala/ML/issues/1148">wala/ML#1148</a>).
 */
public class PythonInstanceKeys extends ZeroXInstanceKeys {

  private final IClassHierarchy cha;

  public PythonInstanceKeys(
      AnalysisOptions options,
      IClassHierarchy cha,
      SSAContextInterpreter contextInterpreter,
      int policy) {
    super(options, cha, contextInterpreter, policy);
    this.cha = cha;
  }

  @Override
  public InstanceKey getInstanceKeyForAllocation(CGNode node, NewSiteReference allocation) {
    if (PythonTypes.isScalar(allocation.getDeclaredType())) {
      IClass type = cha.lookupClass(allocation.getDeclaredType());
      if (type != null) return new ConcreteTypeKey(type);
    }
    return super.getInstanceKeyForAllocation(node, allocation);
  }
}
