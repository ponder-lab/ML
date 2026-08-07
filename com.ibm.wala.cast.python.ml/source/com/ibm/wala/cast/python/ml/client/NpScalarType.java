package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Generator for a call to one of NumPy's scalar types, {@code np.float64(2.0)} and its siblings.
 * The type coerces its argument to its own dtype while preserving the argument's shape, so a Python
 * number in yields a rank-0 array out and a sequence in yields an array of that sequence's shape.
 * The dtype is the type's own and is fixed at dispatch, which is the only way this generator
 * differs from {@link AstypeOperation}'s shape-preserving, dtype-changing contract &mdash; there
 * the target dtype is an argument.
 *
 * <p>The same allocation that this generator is dispatched for also serves as the {@code dtype=}
 * token of the same name; see the module method in {@code numpy.xml} for how one object carries
 * both roles. See <a href="https://github.com/wala/ML/issues/827">wala/ML#827</a>.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/arrays.scalars.html">NumPy scalars</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpScalarType extends TensorGenerator {

  private static final Logger LOGGER = getLogger(NpScalarType.class.getName());

  /** The positional index of the value being coerced. */
  private static final int VALUE_PARAMETER_POSITION = 0;

  /** The dtype this scalar type names, i.e. the dtype of every value it constructs. */
  private final DType dtype;

  /**
   * Constructs a generator anchored to the call's result.
   *
   * @param source The {@link PointsToSetVariable} the call defines.
   * @param dtype The dtype the called scalar type names.
   */
  public NpScalarType(PointsToSetVariable source, DType dtype) {
    super(source);
    this.dtype = dtype;
  }

  /**
   * Constructs a generator anchored to the allocating synthetic node, for producer delegation.
   *
   * @param node The {@link CGNode} for the scalar type's {@code do()} method.
   * @param dtype The dtype the called scalar type names.
   */
  public NpScalarType(CGNode node, DType dtype) {
    super(node);
    this.dtype = dtype;
  }

  /**
   * {@inheritDoc}
   *
   * <p>The shape is the coerced value's: rank 0 for a Python number (the corpus idiom), and the
   * sequence's shape when a sequence is coerced. An argument whose own shape does not resolve
   * leaves the result at ⊤ rather than assuming rank 0, since {@code np.float32([1, 2])} is an
   * array of shape {@code (2,)}, not a scalar.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The coerced value's shapes, or {@code null} (⊤) if they do not resolve.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int valueVn = this.getArgumentValueNumber(VALUE_PARAMETER_POSITION);
    if (valueVn > 0)
      try {
        Set<List<Dimension<?>>> shapes = this.getShapes(builder, this.getNode(), valueVn);
        if (shapes != null && !shapes.isEmpty()) return shapes;
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "Coerced value's shape lookup failed for source: " + describe(this.getSource()) + ".",
            e);
      }

    LOGGER.fine(
        () ->
            "Unresolved coerced value for source: "
                + describe(this.getSource())
                + "; returning ⊤ (unknown shape).");
    return null;
  }

  /**
   * {@inheritDoc}
   *
   * <p>The dtype is the scalar type's own, whatever the argument's was: that coercion is the whole
   * point of the call.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A singleton set containing the dtype this scalar type names.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(this.dtype);
  }

  /**
   * Returns the producing library of the modeled value: a NumPy scalar type constructs an array
   * (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
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
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
