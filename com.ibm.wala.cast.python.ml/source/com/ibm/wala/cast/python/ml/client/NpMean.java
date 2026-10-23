package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for the NumPy averaging reductions {@code np.mean(a, axis=None, dtype=None, out=None,
 * keepdims=False)} and {@code np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)}
 * (<a href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). Without an {@code axis}, the
 * input is reduced over every axis, to a scalar; with a constant integer {@code axis}, that axis is
 * removed, or kept as {@code 1} under {@code keepdims=True}. The dtype is the {@code dtype}
 * argument's when one is given; otherwise an integer or {@code bool} input averages as {@code
 * float64} and a floating one keeps its dtype. A complex input keeps its dtype under {@code mean},
 * and {@code std}, whose result is real, gives the matching floating dtype.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.mean.html">numpy.mean</a>
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.std.html">numpy.std</a>
 */
public class NpMean extends NpArrayFunction {

  /** The positional index of {@code axis}. */
  private static final int AXIS_POSITION = 1;

  /** The positional index of {@code dtype}. */
  private static final int DTYPE_POSITION = 2;

  /** Whether the reduction's result is real, as {@code std}'s is. */
  private final boolean real;

  /**
   * {@inheritDoc}
   *
   * <p>A shallow copy suffices: {@link #real} is a primitive.
   */
  @Override
  protected NpMean clone() throws CloneNotSupportedException {
    return (NpMean) super.clone();
  }

  /**
   * Constructs for a source.
   *
   * @param source The source.
   * @param real Whether the reduction's result is real, as {@code std}'s is.
   */
  public NpMean(PointsToSetVariable source, boolean real) {
    super(source);
    this.real = real;
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   * @param real Whether the reduction's result is real, as {@code std}'s is.
   */
  public NpMean(CGNode node, boolean real) {
    super(node);
    this.real = real;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // `np.std` has `ddof` before `keepdims`.
    int keepDimsPosition = this.real ? 5 : 4;
    Boolean keepDims = Boolean.FALSE;
    if (this.isArgumentPassed(builder, keepDimsPosition, "keepdims")) {
      keepDims = this.getConstantBooleanArgument(builder, keepDimsPosition, "keepdims");
      if (keepDims == null) return null;
    }
    boolean all = !this.isArgumentPassed(builder, AXIS_POSITION, "axis");
    Integer axis = all ? null : this.getConstantIntegerArgument(builder, AXIS_POSITION, "axis");
    if (!all && axis == null) return null;

    Set<List<Dimension<?>>> inputs = this.getShapesOfArgument(builder, 0);
    if (inputs == null) return all && !keepDims ? Collections.singleton(List.of()) : null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputs) {
      int rank = input.size();
      int reduced = all ? -1 : axis < 0 ? axis + rank : axis;
      if (!all && (reduced < 0 || reduced >= rank)) continue;
      List<Dimension<?>> output = new ArrayList<>(rank);
      for (int i = 0; i < rank; i++)
        if (all || i == reduced) {
          if (keepDims) output.add(new NumericDim(1));
        } else output.add(input.get(i));
      ret.add(output);
    }
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> dtypes =
        this.getArgumentPointsToSet(builder, DTYPE_POSITION, this.getDTypeParameterName());
    if (dtypes != null && !dtypes.isEmpty() && !allNullConstants(dtypes)) {
      Set<DType> given = this.getDTypesFromDTypeArgument(builder, dtypes);
      if (!given.isEmpty()) return given;
    }
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (DType input : this.getNumpyDTypesOfArgument(builder, 0)) ret.add(this.mapDType(input));
    return this.dTypeApiDefaultOrUnknown(builder, ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The dtype NumPy averages an input dtype in.
   */
  @Override
  protected DType mapDType(DType input) {
    switch (input) {
      case BOOL:
      case UINT8:
      case INT32:
      case INT64:
      case FLOAT64:
        return DType.FLOAT64;
      case FLOAT32:
        return DType.FLOAT32;
      case COMPLEX64:
        return this.real ? DType.FLOAT32 : input;
      case COMPLEX128:
        return this.real ? DType.FLOAT64 : input;
      default:
        return DType.UNKNOWN;
    }
  }

  /**
   * The boolean constant an argument always is, a Python {@code bool} or an integer read as one.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The argument's positional index, excluding the callable.
   * @param name The argument's keyword name.
   * @return The constant, or {@code null} when the argument is not one boolean constant.
   */
  private Boolean getConstantBooleanArgument(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    if (pts == null || pts.isEmpty()) return null;
    Boolean ret = null;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey<?> constant)) return null;
      Boolean value =
          constant.getValue() instanceof Boolean b
              ? b
              : constant.getValue() instanceof Integer i ? i != 0 : null;
      if (value == null || (ret != null && !ret.equals(value))) return null;
      ret = value;
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * <p>No feed when a {@code dtype} argument is passed: a readable one proves the seed's dtype, so
   * there is nothing to feed, and an unreadable one leaves the dtype unknown (wala/ML#865), which a
   * feed mapping the input's dtype would overwrite with one the argument may not name.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    if (this.isArgumentPassed(builder, DTYPE_POSITION, this.getDTypeParameterName())) return null;
    return super.getTypeFeed(builder);
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
