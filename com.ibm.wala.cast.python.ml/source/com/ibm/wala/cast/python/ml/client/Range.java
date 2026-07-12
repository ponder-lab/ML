package com.ibm.wala.cast.python.ml.client;

import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the TensorFlow range operation.
 *
 * <p>This class is used to generate a tensor that contains a sequence of numbers, similar to the
 * range function in Python.
 *
 * <p>Output dtype follows {@code tf.range}'s runtime behaviour: an explicit {@code dtype=} keyword
 * is honoured; otherwise the dtype is derived from the {@code start}/{@code limit}/{@code delta}
 * argument types (any float operand promotes the result to {@code float32}), defaulting to {@code
 * int32} when nothing more precise is available. Fix for <a
 * href="https://github.com/wala/ML/issues/492">wala/ML#492</a>.
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
      return name().toLowerCase(Locale.ROOT);
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

    // 1. Precise Caller Resolution: walk the call-graph edges to the invocations dispatching to
    // this node.
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode()))
      if (callerInvoke.snd instanceof PythonInvokeInstruction)
        ret.addAll(
            processCall(builder, callerInvoke.fst, (PythonInvokeInstruction) callerInvoke.snd));

    // 2. Fallback for non-CS or if CS failed

    if (ret.isEmpty()) {
      for (Integer numOfPoisitionArguments : this.getNumberOfPossiblePositionalArguments(builder)) {
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
            limitPts =
                this.getArgumentPointsToSet(
                    builder, getStartParameterPosition(), getLimitParameterName());
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
        Set<Double> limits = getPossibleDoubleValues(limitPts);
        Set<Double> deltas = getPossibleDoubleValues(deltaPts);

        // A null set means the argument is not statically resolvable (wala/ML#669); with no
        // resolvable `limit`, no concrete shape is produced below.
        if (starts == null) starts = HashSetFactory.make();
        if (limits == null) limits = HashSetFactory.make();
        if (deltas == null) deltas = HashSetFactory.make();

        if (starts.isEmpty()) starts.add(0.0);
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

    // No bound combination resolved. The call still produces a rank-1 tensor whose fixed length
    // the analysis could not compute, so fall back to the unresolved-length default rather than
    // returning the empty set — ⊥ ("not a tensor") silently drops the variable (wala/ML#721).
    return ret.isEmpty() ? this.getDefaultShapes(builder) : ret;
  }

  /**
   * Computes the possible shapes of the tensor produced by a specific call to the {@code range}
   * function.
   *
   * <p>This method analyzes the {@code start}, {@code limit}, and {@code delta} arguments from a
   * concrete {@link PythonInvokeInstruction} to determine the size of the resulting 1D tensor. It
   * is primarily used during context-sensitive analysis when the exact call instruction is known,
   * allowing for more precise argument resolution compared to the general {@link #getShapes}
   * fallback.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param caller The {@link CGNode} calling the function.
   * @param pyCallInstr The {@link PythonInvokeInstruction} representing the call to {@code range}.
   * @return A set of shapes, where each shape is represented as a list of dimensions.
   */
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
      if (limitVN == UNDEFINED_PARAMETER_POSITION) {
        // tf.range(X) -> limit=X
        limitVN = pyCallInstr.getUse(1);
      } else {
        // tf.range(X, limit=Y) -> start=X, limit=Y
        if (startVN == UNDEFINED_PARAMETER_POSITION) startVN = pyCallInstr.getUse(1);
      }
    } else if (numPosArgs == 3) { // 2 positional args
      if (startVN == UNDEFINED_PARAMETER_POSITION) startVN = pyCallInstr.getUse(1);
      if (limitVN == UNDEFINED_PARAMETER_POSITION) limitVN = pyCallInstr.getUse(2);
    } else if (numPosArgs >= 4) { // 3+ positional args
      if (startVN == UNDEFINED_PARAMETER_POSITION) startVN = pyCallInstr.getUse(1);
      if (limitVN == UNDEFINED_PARAMETER_POSITION) limitVN = pyCallInstr.getUse(2);
      if (deltaVN == UNDEFINED_PARAMETER_POSITION) deltaVN = pyCallInstr.getUse(3);
    }

    // Special case for keyword-only: tf.range(start=5) -> limit=5, start=0.
    if (numPosArgs == 1) { // only function object
      if (limitVN == UNDEFINED_PARAMETER_POSITION && startVN != UNDEFINED_PARAMETER_POSITION) {
        limitVN = startVN;
        startVN = UNDEFINED_PARAMETER_POSITION;
      }
    }

    Set<Double> starts = getPossibleDoubleValues(builder, caller, startVN);
    Set<Double> limits = getPossibleDoubleValues(builder, caller, limitVN);
    Set<Double> deltas = getPossibleDoubleValues(builder, caller, deltaVN);

    // A null set means the argument is not statically resolvable (wala/ML#669); with no
    // resolvable `limit`, no concrete shape is produced below.
    if (starts == null) starts = HashSetFactory.make();
    if (limits == null) limits = HashSetFactory.make();
    if (deltas == null) deltas = HashSetFactory.make();

    if (starts.isEmpty()) starts.add(0.0);
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

  /**
   * Derives the output dtype from the numeric {@code start}/{@code limit}/{@code delta} arguments
   * when no explicit {@code dtype=} keyword is supplied, matching {@code tf.range}'s runtime
   * promotion rule: any float operand promotes the result to {@code float32}, otherwise it stays
   * {@code int32}. Falls back to {@code int32} when nothing about the operand types is recoverable
   * (the conservative default at runtime).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype set derived from the numeric arguments, or {@code int32} as a fallback.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> argPts = OrdinalSet.empty();
    argPts =
        OrdinalSet.unify(
            argPts,
            this.getArgumentPointsToSet(
                builder, getStartParameterPosition(), getStartParameterName()));
    argPts =
        OrdinalSet.unify(
            argPts,
            this.getArgumentPointsToSet(
                builder, getLimitParameterPosition(), getLimitParameterName()));
    argPts =
        OrdinalSet.unify(
            argPts,
            this.getArgumentPointsToSet(
                builder, getDeltaParameterPosition(), getDeltaParameterName()));

    if (argPts.isEmpty()) return EnumSet.of(DType.INT32);

    Set<DType> derived = this.getDTypesOfValue(builder, argPts);
    if (derived == null || derived.isEmpty()) return EnumSet.of(DType.INT32);

    // TF's runtime promotion: any float operand promotes the entire result to float32, dropping
    // integer dtypes from the mix. E.g., `tf.range(0, 5.0)` → float32, not {INT32, FLOAT32}.
    if (derived.contains(DType.FLOAT32) || derived.contains(DType.FLOAT64))
      return EnumSet.of(DType.FLOAT32);
    return derived;
  }

  /**
   * Floors an unresolvable {@code tf.range} to a rank-1 tensor with a dynamic length. The precise
   * length is computed in {@link #getShapes} from the numeric arguments; this fallback is reached
   * only when those arguments are unresolvable (content-dependent). {@code tf.range} always
   * produces a rank-1 tensor, so the rank is still known: floor to a single {@link DynamicDim} axis
   * rather than throwing (which aborted the whole analysis) or dropping to ⊤. See <a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>, mirroring the floors in <a
   * href="https://github.com/wala/ML/issues/604">wala/ML#604</a>/<a
   * href="https://github.com/wala/ML/issues/606">wala/ML#606</a>.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A single rank-1 shape with a {@link DynamicDim} length.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    List<Dimension<?>> rank1 = new ArrayList<>();
    // Unresolvable bounds are typically Python scalars, so the length is a fixed runtime value
    // the analysis could not compute (wala/ML#721).
    rank1.add(UnresolvedDim.INSTANCE);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(rank1);
    return ret;
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * Returns the position of the start parameter.
   *
   * @return the position of the start parameter
   */
  protected static int getStartParameterPosition() {
    return Parameters.START.getIndex();
  }

  /**
   * Returns the position of the limit parameter.
   *
   * @return the position of the limit parameter
   */
  protected static int getLimitParameterPosition() {
    return Parameters.LIMIT.getIndex();
  }

  /**
   * Returns the position of the delta parameter.
   *
   * @return the position of the delta parameter
   */
  protected static int getDeltaParameterPosition() {
    return Parameters.DELTA.getIndex();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  /**
   * Returns the name of the start parameter.
   *
   * @return the name of the start parameter
   */
  protected static String getStartParameterName() {
    return Parameters.START.getName();
  }

  /**
   * Returns the name of the limit parameter.
   *
   * @return the name of the limit parameter
   */
  protected static String getLimitParameterName() {
    return Parameters.LIMIT.getName();
  }

  /**
   * Returns the name of the delta parameter.
   *
   * @return the name of the delta parameter
   */
  protected static String getDeltaParameterName() {
    return Parameters.DELTA.getName();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
