package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
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

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Integer bounds draw numpy's default integer dtype; bounds the analysis cannot read may be
    // floats, whose result would be float64, so the dtype is unknown rather than a guess.
    boolean allIntegral = true;
    boolean any = false;
    for (Pair<CGNode, PythonInvokeInstruction> callerInvoke : this.callerInvokes(builder)) {
      any = true;
      for (int vn : boundValueNumbers(callerInvoke.snd))
        if (vn >= 0 && resolve(builder, callerInvoke.fst, vn) == null) allIntegral = false;
    }
    Set<DType> operandDefault =
        any && allIntegral ? EnumSet.of(DType.INT64) : EnumSet.of(DType.UNKNOWN);
    return this.dTypeApiDefaultOrUnknown(builder, operandDefault);
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
}
