package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code np.arange([start,] stop[, step], dtype=None)} (wala/ML#909): a rank-1 array
 * of {@code ceil((stop - start) / step)} elements. The extent is concrete when the bounds resolve
 * to integer constants and {@link UnresolvedDim} otherwise, since an unresolvable bound is a fixed
 * runtime integer the analysis could not compute (wala/ML#721); {@code tf.range} floors the same
 * way. A padding that follows can still fold the bounds as terms, which is {@link NpPad}'s chase,
 * not this generator's: this one resolves values only.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/generated/numpy.arange.html">numpy.arange</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpArange extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(NpArange.class.getName());

  /**
   * The 0-based positional index of {@code dtype}, after {@code start}, {@code stop} and {@code
   * step}.
   */
  private static final int DTYPE_POSITION = 3;

  public NpArange(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public NpArange(CGNode node) {
    super(node);
  }

  /**
   * The bound value numbers of an {@code arange} call in its caller's frame, in numpy's positional
   * convention: one positional argument is {@code stop}, two are {@code start, stop}, three add
   * {@code step}; keywords override.
   *
   * @param call The {@code arange} invoke.
   * @return {@code start}, {@code stop} and {@code step} value numbers, {@code -1} where absent.
   */
  static int[] boundValueNumbers(PythonInvokeInstruction call) {
    int start = call.getUse("start");
    int stop = call.getUse("stop");
    int step = call.getUse("step");
    int positional = call.getNumberOfPositionalParameters();
    if (positional == 2 && stop < 0) stop = call.getUse(1);
    else if (positional == 2 && start < 0) start = call.getUse(1);
    if (positional >= 3) {
      if (start < 0) start = call.getUse(1);
      if (stop < 0) stop = call.getUse(2);
    }
    if (positional >= 4 && step < 0) step = call.getUse(3);
    return new int[] {start, stop, step};
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Pair<CGNode, PythonInvokeInstruction> callerInvoke : this.callerInvokes(builder)) {
      CGNode caller = callerInvoke.fst;
      int[] bounds = boundValueNumbers(callerInvoke.snd);
      Integer start = bounds[0] < 0 ? Integer.valueOf(0) : resolve(builder, caller, bounds[0]);
      Integer stop = resolve(builder, caller, bounds[1]);
      Integer step = bounds[2] < 0 ? Integer.valueOf(1) : resolve(builder, caller, bounds[2]);
      List<Dimension<?>> shape = new ArrayList<>();
      if (start == null || stop == null || step == null || step == 0)
        shape.add(UnresolvedDim.INSTANCE);
      else shape.add(new NumericDim(Math.max(0, ceilDiv(stop - start, step))));
      ret.add(shape);
    }
    if (ret.isEmpty()) {
      // No readable call site: the rank is still one.
      ret.add(List.of(UnresolvedDim.INSTANCE));
    }
    LOGGER.fine(() -> "np.arange resolved to " + ret + ".");
    return ret;
  }

  /**
   * The call sites this generator reads its arguments from: the anchoring invoke when the source is
   * anchored in a caller frame, else every reachable invoke of the anchored node.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The (caller, invoke) pairs.
   */
  private List<Pair<CGNode, PythonInvokeInstruction>> callerInvokes(
      PropagationCallGraphBuilder builder) {
    List<Pair<CGNode, PythonInvokeInstruction>> ret = new ArrayList<>();
    PythonInvokeInstruction own = this.getInvokeInstruction();
    if (own != null) ret.add(Pair.make(this.getNode(), own));
    else
      for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
          getCallerInvokes(builder, this.getNode()))
        if (callerInvoke.snd instanceof PythonInvokeInstruction)
          ret.add(Pair.make(callerInvoke.fst, (PythonInvokeInstruction) callerInvoke.snd));
    return ret;
  }

  private static Integer resolve(PropagationCallGraphBuilder builder, CGNode node, int vn) {
    return vn < 0
        ? null
        : resolveIntFlowSensitively(
            builder, node, vn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
  }

  private static int ceilDiv(int numerator, int denominator) {
    return (int) Math.ceil((double) numerator / denominator);
  }

  /**
   * {@inheritDoc}
   *
   * <p>NumPy draws the dtype from the bounds: integers give its default integer, {@code int64}, and
   * a float among them gives {@code float64}. A bound's value is not enough to tell the two apart,
   * since {@code -4.0} and {@code -4} resolve to the same number, so each bound's numeric kind is
   * read from how it is written ({@link #numericKind}). A quotient of integers is either: the front
   * end translates {@code /} and {@code //} to one operator, so such a bound gives both dtypes, the
   * sound may-set. A bound whose kind cannot be read gives an unknown dtype (wala/ML#1008).
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    boolean any = false;
    boolean unknown = false;
    boolean anyFloat = false;
    boolean ambiguous = false;
    for (Pair<CGNode, PythonInvokeInstruction> callerInvoke : this.callerInvokes(builder)) {
      any = true;
      for (int vn : boundValueNumbers(callerInvoke.snd)) {
        if (vn < 0) continue;
        NumericKind kind =
            numericKind(builder, callerInvoke.fst, vn, new HashSet<>(), NUMERIC_KIND_DEPTH_CAP);
        if (kind == null) unknown = true;
        else if (kind == NumericKind.FLOAT) anyFloat = true;
        else if (kind == NumericKind.INT_OR_FLOAT) ambiguous = true;
      }
    }
    Set<DType> operandDefault;
    if (!any || unknown) operandDefault = EnumSet.of(DType.UNKNOWN);
    else if (anyFloat) operandDefault = EnumSet.of(DType.FLOAT64);
    else if (ambiguous) operandDefault = EnumSet.of(DType.INT64, DType.FLOAT64);
    else operandDefault = EnumSet.of(DType.INT64);
    return this.dTypeApiDefaultOrUnknown(builder, operandDefault);
  }

  /** How deep {@link #numericKind} follows a bound's definitions. */
  private static final int NUMERIC_KIND_DEPTH_CAP = 8;

  /** The numeric kind of a Python number as its writing determines it. */
  enum NumericKind {
    /** An {@code int} (or {@code bool}). */
    INT,
    /** A {@code float}. */
    FLOAT,
    /** One of the two, which the writing does not decide. */
    INT_OR_FLOAT;

    private NumericKind join(NumericKind other) {
      return this == other ? this : INT_OR_FLOAT;
    }
  }

  /**
   * The numeric kind of a value as its writing determines it: a literal's own kind, an arithmetic
   * result's from its operands' (a {@code float} operand makes it a {@code float}, and a quotient
   * of integers is either kind, since {@code /} and {@code //} share an operator), a negation's
   * from its operand's, and a lexical read's from the writes that reach it. A value resolving to a
   * number by another route is either kind.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node whose IR defines the value.
   * @param vn The value number.
   * @param visited The (node, value number) pairs already followed.
   * @param depth The remaining depth.
   * @return The kind, or {@code null} when the value is not known to be a number.
   */
  static NumericKind numericKind(
      PropagationCallGraphBuilder builder,
      CGNode node,
      int vn,
      Set<Pair<CGNode, Integer>> visited,
      int depth) {
    if (vn <= 0 || depth <= 0 || node.getIR() == null || node.getDU() == null) return null;
    if (!visited.add(Pair.make(node, vn))) return null;
    SymbolTable symbols = node.getIR().getSymbolTable();
    if (symbols.isConstant(vn)) {
      Object value = symbols.getConstantValue(vn);
      if (value instanceof Integer || value instanceof Long || value instanceof Boolean)
        return NumericKind.INT;
      if (value instanceof Double || value instanceof Float) return NumericKind.FLOAT;
      return null;
    }
    SSAInstruction def = node.getDU().getDef(vn);
    if (def instanceof SSABinaryOpInstruction op) {
      NumericKind left = numericKind(builder, node, op.getUse(0), visited, depth - 1);
      NumericKind right = numericKind(builder, node, op.getUse(1), visited, depth - 1);
      if (left == null || right == null) return null;
      if (left == NumericKind.FLOAT || right == NumericKind.FLOAT) return NumericKind.FLOAT;
      IBinaryOpInstruction.IOperator operator = op.getOperator();
      if (operator == IBinaryOpInstruction.Operator.ADD
          || operator == IBinaryOpInstruction.Operator.SUB
          || operator == IBinaryOpInstruction.Operator.MUL
          || operator == IBinaryOpInstruction.Operator.REM) return left.join(right);
      return NumericKind.INT_OR_FLOAT;
    }
    if (def instanceof SSAUnaryOpInstruction op)
      return numericKind(builder, node, op.getUse(0), visited, depth - 1);
    if (def instanceof AstLexicalRead) {
      NumericKind ret = null;
      for (Pair<CGNode, Integer> definer : lexicalDefiners(builder, node, vn)) {
        if (definer.snd > 0
            && definer.fst.getDU() != null
            && definer.fst.getDU().getDef(definer.snd) instanceof SSAPhiInstruction) continue;
        NumericKind kind = numericKind(builder, definer.fst, definer.snd, visited, depth - 1);
        if (kind == null) return null;
        ret = ret == null ? kind : ret.join(kind);
      }
      if (ret != null) return ret;
    }
    if (resolveIntFlowSensitively(
            builder, node, vn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP)
        == null) return null;
    return def instanceof PythonInvokeInstruction call && callsIntBuiltin(builder, node, call)
        ? NumericKind.INT
        : NumericKind.INT_OR_FLOAT;
  }

  /**
   * Whether a call's only target is the {@code int} builtin, whose result is an {@code int}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph lookup.
   * @param node The node whose IR holds the call.
   * @param call The call.
   * @return {@code true} iff the call can only be {@code int(...)}.
   */
  private static boolean callsIntBuiltin(
      PropagationCallGraphBuilder builder, CGNode node, PythonInvokeInstruction call) {
    Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, call.getCallSite());
    return targets != null
        && targets.size() == 1
        && "Lwala/builtin/int"
            .equals(
                targets
                    .iterator()
                    .next()
                    .getMethod()
                    .getDeclaringClass()
                    .getReference()
                    .getName()
                    .toString());
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
    return DTYPE_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }

  /**
   * Returns the producing library of the modeled value: an {@code np.arange(...)} call, so the
   * value is an ndarray (wala/ML#724, wala/ML#980).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
