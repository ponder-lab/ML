package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for ndarrays created by NumPy's {@code full_like()} function.
 *
 * <p>The result has the prototype {@code a}'s shape and dtype. An explicit {@code shape} argument
 * replaces the shape and an explicit {@code dtype} argument the dtype; the fill value never decides
 * the dtype ({@code np.full_like(int32_array, 0.5)} is an {@code int32} array). The prototype's
 * dtype is read with NumPy's conversion, so a Python scalar prototype is {@code int64} or {@code
 * float64}, as in {@link NpFull}.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/generated/numpy.full_like.html">numpy.full_like()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpFullLike extends ValueExtractingTensorGenerator {

  private static final Logger LOGGER = getLogger(NpFullLike.class.getName());

  /** The parameters of {@code np.full_like}, in positional order. */
  protected enum Parameters {
    A,
    FILL_VALUE,
    DTYPE,
    ORDER,
    SUBOK,
    SHAPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public NpFullLike(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public NpFullLike(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * <p>An explicit {@code shape=None} means "use the prototype's shape", so it takes the default
   * path rather than being read as a shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> shape =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());
    if (shape != null && !shape.isEmpty() && allNullConstants(shape))
      return this.getDefaultShapes(builder);
    return super.getShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Without a {@code dtype} argument, the dtype is the prototype's, read with NumPy's conversion
   * (see {@link NpFull#numpyDTypesOfValue}). A supplied {@code dtype} that nothing could read
   * degrades to {@link DType#UNKNOWN} rather than borrowing the prototype's dtype (wala/ML#865).
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return The prototype's NumPy dtypes, or {@link DType#UNKNOWN}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> prototype =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    // A Python scalar built by the `float`, `int` or `bool` builtin has an empty points-to set;
    // NumPy converts it as a literal of its type.
    DType scalarBuiltin =
        prototype == null || prototype.isEmpty()
            ? this.getArgumentScalarBuiltinDType(
                builder, this.getValueParameterPosition(), this.getValueParameterName())
            : null;
    Set<DType> ret =
        this.dTypeApiDefaultOrUnknown(
            builder,
            scalarBuiltin != null
                ? EnumSet.of(numpyDTypeOfPythonScalar(scalarBuiltin))
                : NpFull.numpyDTypesOfValue(this, builder, prototype));

    LOGGER.fine(() -> "Default dtypes for source: " + describe(source) + " are: " + ret + ".");

    return ret;
  }

  @Override
  protected int getValueParameterPosition() {
    return Parameters.A.getIndex();
  }

  @Override
  protected String getValueParameterName() {
    return Parameters.A.getName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  /**
   * Returns the producing library of the modeled value: an {@code np.full_like(...)} call, so the
   * value is an ndarray (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
