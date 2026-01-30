package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContext;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/** A representation of the TensorFlow range operation. */
public class Range extends TensorGenerator {

  protected enum Parameters {
    START,
    LIMIT,
    DELTA,
    DTYPE,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  @SuppressWarnings("unused")
  private static final Logger LOGGER = Logger.getLogger(Range.class.getName());

  public Range(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // 1. Precise Context-Sensitive Resolution
    Context cs = this.getNode().getContext();
    if (cs instanceof CallStringContext) {
      CallStringContext csc = (CallStringContext) cs;
      CallString callString = (CallString) csc.get(CALL_STRING);
      CallSiteReference[] sites = callString.getCallSiteRefs();
      IMethod[] methods = callString.getMethods();

      if (sites.length > 0 && methods.length > 0) {
        CallSiteReference siteReference = sites[sites.length - 1];
        IMethod callerMethod = methods[methods.length - 1];

        Iterator<CGNode> preds = builder.getCallGraph().getPredNodes(this.getNode());
        while (preds.hasNext()) {
          CGNode caller = preds.next();
          if (!caller.getMethod().equals(callerMethod)) continue;

          SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);
          for (SSAAbstractInvokeInstruction callInstr : calls) {
            if (callInstr.getCallSite().equals(siteReference)
                && callInstr instanceof PythonInvokeInstruction) {
              processCall(builder, caller, (PythonInvokeInstruction) callInstr, ret);
            }
          }
        }
      }
    }

    // 2. Fallback for non-CS or if CS failed
    if (ret.isEmpty()) {
      for (Integer numOfPoisitionArguments : getNumberOfPossiblePositionalArguments(builder)) {
        OrdinalSet<InstanceKey> startPts =
            this.getArgumentPointsToSet(builder, 0, Parameters.START.getName());
        OrdinalSet<InstanceKey> limitPts =
            this.getArgumentPointsToSet(builder, 1, Parameters.LIMIT.getName());
        OrdinalSet<InstanceKey> deltaPts =
            this.getArgumentPointsToSet(builder, 2, Parameters.DELTA.getName());

        if (numOfPoisitionArguments == 1
            && !isKeywordArgumentPresent(builder, Parameters.LIMIT.getName())) {
          limitPts = this.getArgumentPointsToSet(builder, 0, null);
          startPts = OrdinalSet.empty();
        }

        Set<Double> starts = getPossibleDoubleValues(startPts);
        if (starts.isEmpty()) starts.add(0.0);
        Set<Double> limits = getPossibleDoubleValues(limitPts);
        Set<Double> deltas = getPossibleDoubleValues(deltaPts);
        if (deltas.isEmpty()) deltas.add(1.0);

        for (Double s : starts) {
          for (Double l : limits) {
            for (Double d : deltas) {
              ret.add(List.of(new NumericDim((int) Math.ceil((l - s) / d))));
            }
          }
        }
      }
    }

    return ret;
  }

  private void processCall(
      PropagationCallGraphBuilder builder,
      CGNode caller,
      PythonInvokeInstruction pyCallInstr,
      Set<List<Dimension<?>>> ret) {
    int numPosArgs = pyCallInstr.getNumberOfPositionalParameters();

    int startVN = pyCallInstr.getUse(Parameters.START.getName());
    int limitVN = pyCallInstr.getUse(Parameters.LIMIT.getName());
    int deltaVN = pyCallInstr.getUse(Parameters.DELTA.getName());

    // Positional assignment (index 0 is the function object)
    if (numPosArgs == 2) { // range(limit)
      if (limitVN == -1) limitVN = pyCallInstr.getUse(1);
    } else if (numPosArgs == 3) { // range(start, limit)
      if (startVN == -1) startVN = pyCallInstr.getUse(1);
      if (limitVN == -1) limitVN = pyCallInstr.getUse(2);
    } else if (numPosArgs >= 4) { // range(start, limit, delta)
      if (startVN == -1) startVN = pyCallInstr.getUse(1);
      if (limitVN == -1) limitVN = pyCallInstr.getUse(2);
      if (deltaVN == -1) deltaVN = pyCallInstr.getUse(3);
    }

    Set<Double> starts = getPossibleDoubleValues(builder, caller, startVN);
    if (starts.isEmpty()) starts.add(0.0);
    Set<Double> limits = getPossibleDoubleValues(builder, caller, limitVN);
    Set<Double> deltas = getPossibleDoubleValues(builder, caller, deltaVN);
    if (deltas.isEmpty()) deltas.add(1.0);

    for (Double s : starts) {
      for (Double l : limits) {
        for (Double d : deltas) {
          ret.add(List.of(new NumericDim((int) Math.ceil((l - s) / d))));
        }
      }
    }
  }

  private Set<Double> getPossibleDoubleValues(
      PropagationCallGraphBuilder builder, CGNode caller, int vn) {
    if (vn == -1) return HashSetFactory.make();
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, vn);
    return getPossibleDoubleValues(builder.getPointerAnalysis().getPointsToSet(pk));
  }

  private Set<Double> getPossibleDoubleValues(OrdinalSet<InstanceKey> pts) {
    Set<Double> vals = HashSetFactory.make();
    if (pts != null) {
      for (InstanceKey ik : pts) {
        if (ik instanceof ConstantKey) {
          Object val = ((ConstantKey<?>) ik).getValue();
          if (val instanceof Number) vals.add(((Number) val).doubleValue());
        }
      }
    }
    return vals;
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.INT32);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException("Shapes derived from arguments.");
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
