package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.logging.Logger.getLogger;

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

/**
 * A representation of the TensorFlow range operation.
 *
 * <p>This class is used to generate a tensor that contains a sequence of numbers, similar to the
 * range function in Python.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/range">TensorFlow range
 *     documentation</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
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
  private static final Logger LOGGER = getLogger(Range.class.getName());

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
              ret.addAll(processCall(builder, caller, (PythonInvokeInstruction) callInstr));
            }
          }
        }
      }
    }

    // 2. Fallback for non-CS or if CS failed
    if (ret.isEmpty()) {
      for (Integer numOfPoisitionArguments : getNumberOfPossiblePositionalArguments(builder)) {
        OrdinalSet<InstanceKey> startPts =
            this.getArgumentPointsToSet(
                builder, getStartParameterPosition(), getStartParameterName());
        OrdinalSet<InstanceKey> limitPts =
            this.getArgumentPointsToSet(
                builder, getLimitParameterPosition(), getLimitParameterName());
        OrdinalSet<InstanceKey> deltaPts =
            this.getArgumentPointsToSet(
                builder, getDeltaParameterPosition(), getDeltaParameterName());

        if (numOfPoisitionArguments == 0) {
          // All keywords.
          // Note: tf.range(start=5) is valid (behaves as range(limit=5)).
          // Note: tf.range(limit=5) is invalid.
          if (!this.isKeywordArgumentPresent(builder, getStartParameterName())) {
            throw new IllegalStateException(
                "Expected at least 'start' keyword when 0 positional arguments are provided for"
                    + " range().");
          }

          if (!this.isKeywordArgumentPresent(builder, getLimitParameterName())) {
            // tf.range(start=5) -> limit=5, start=0.
            limitPts = startPts;
            startPts = OrdinalSet.empty();
          }
        } else if (numOfPoisitionArguments == 1) {
          // 1. tf.range(limit) -> start=0, delta=1
          // OR tf.range(start, limit=X) -> start=pos0, limit=X
          if (!this.isKeywordArgumentPresent(builder, getLimitParameterName())) {
            limitPts = this.getArgumentPointsToSet(builder, getStartParameterPosition(), null);
            startPts = OrdinalSet.empty();
          }
          // Note: if limit keyword is present, startPts already contains pos 0 and limitPts already
          // contains the keyword. Correct.
        } else if (numOfPoisitionArguments == 2) {
          // 2. tf.range(start, limit, delta=1)
          // No special handling needed; arguments are retrieved by getArgumentPointsToSet.
        } else if (numOfPoisitionArguments >= 3) {
          // 3. tf.range(start, limit, delta)
          // No special handling needed; arguments are retrieved by getArgumentPointsToSet.
        } else {
          throw new IllegalStateException(
              "Invalid argument combination for range(): "
                  + numOfPoisitionArguments
                  + " positional arguments. Expected 1, 2, or >=3 positional arguments, or 0"
                  + " positional arguments with valid keywords.");
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

  private static Set<List<Dimension<?>>> processCall(
      PropagationCallGraphBuilder builder, CGNode caller, PythonInvokeInstruction pyCallInstr) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    int numPosArgs = pyCallInstr.getNumberOfPositionalParameters();

    int startVN = pyCallInstr.getUse(getStartParameterName());
    int limitVN = pyCallInstr.getUse(getLimitParameterName());
    int deltaVN = pyCallInstr.getUse(getDeltaParameterName());

    // Assign positional parameters based on presence of keywords and count.
    // Index 0 is the function object.
    if (numPosArgs == 2) { // 1 positional arg
      if (limitVN == -1) {
        // tf.range(X) -> limit=X
        limitVN = pyCallInstr.getUse(1);
      } else {
        // tf.range(X, limit=Y) -> start=X, limit=Y
        if (startVN == -1) startVN = pyCallInstr.getUse(1);
      }
    } else if (numPosArgs == 3) { // 2 positional args
      if (startVN == -1) startVN = pyCallInstr.getUse(1);
      if (limitVN == -1) limitVN = pyCallInstr.getUse(2);
    } else if (numPosArgs >= 4) { // 3+ positional args
      if (startVN == -1) startVN = pyCallInstr.getUse(1);
      if (limitVN == -1) limitVN = pyCallInstr.getUse(2);
      if (deltaVN == -1) deltaVN = pyCallInstr.getUse(3);
    }

    // Special case for keyword-only: tf.range(start=5) -> limit=5, start=0.
    if (numPosArgs == 1) { // only function object
      if (limitVN == -1 && startVN != -1) {
        limitVN = startVN;
        startVN = -1;
      }
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
    return ret;
  }

  private static Set<Double> getPossibleDoubleValues(
      PropagationCallGraphBuilder builder, CGNode caller, int vn) {
    Set<Double> vals = HashSetFactory.make();
    if (vn == -1) return vals;

    // 1. Try symbol table (for literal constants)
    if (caller.getIR().getSymbolTable().isConstant(vn)) {
      Object val = caller.getIR().getSymbolTable().getConstantValue(vn);
      if (val instanceof Number) {
        vals.add(((Number) val).doubleValue());
      }
    }

    // 2. Try points-to analysis
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, vn);
    vals.addAll(getPossibleDoubleValues(builder.getPointerAnalysis().getPointsToSet(pk)));

    return vals;
  }

  private static Set<Double> getPossibleDoubleValues(OrdinalSet<InstanceKey> pts) {
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
    throw new UnsupportedOperationException(
        "Shapes for range() are derived from mandatory numeric arguments.");
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  protected static int getStartParameterPosition() {
    return Parameters.START.getIndex();
  }

  protected static int getLimitParameterPosition() {
    return Parameters.LIMIT.getIndex();
  }

  protected static int getDeltaParameterPosition() {
    return Parameters.DELTA.getIndex();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  protected static String getStartParameterName() {
    return Parameters.START.getName();
  }

  protected static String getLimitParameterName() {
    return Parameters.LIMIT.getName();
  }

  protected static String getDeltaParameterName() {
    return Parameters.DELTA.getName();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
