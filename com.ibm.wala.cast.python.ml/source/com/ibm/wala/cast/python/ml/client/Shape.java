package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.shape}. The result is a rank-1 {@code int32} tensor whose values are the
 * runtime shape of {@code input}, so its own shape is {@code (r,)} where {@code r} is the operand's
 * rank. That single rule covers every operand the API accepts: a tensor of rank {@code r}, an
 * array-like converted to rank {@code r} (a flat list is rank 1), and a scalar, whose shape vector
 * is empty, {@code (0,)}. There is no non-tensor result and hence no ⊥ arm: the API converts
 * whatever it is given (wala/ML#943).
 *
 * <p>The extent is read from the operand's resolved shapes, so two distinct readings must not be
 * confused: an operand whose rank is unknown ({@code null} dimensions) yields {@code [Unresolved]},
 * while a scalar operand (empty dimensions) yields {@code (0,)}. Collapsing the first into the
 * second would assert a concrete extent for a rank the analysis never saw, the confidently-wrong
 * class. The unresolved case is {@code Unresolved} rather than {@code Dynamic} under the
 * wala/ML#721 criterion: the extent is a fixed runtime integer the analysis could not compute, and
 * the analysis's ignorance of the rank is not {@code None}-evidence. When the operand's rank is
 * genuinely undeclared to TensorFlow, {@code tf.shape}'s own static shape is {@code (None,)} and
 * the axis would be {@code Dynamic}; that case is not distinguishable from the analysis's ignorance
 * here, so the two collapse to {@code Unresolved}. An array-like operand whose rank the operand
 * reader cannot recover also reads {@code [Unresolved]}, the sound direction.
 *
 * <p>The values of the vector, as opposed to its shape, are read by the shape-vector machinery
 * ({@code TensorGenerator#dispatchesToTfShape}, wala/ML#722), which dispatches on call-graph
 * targets and not on this result's type, so typing the result here does not change what a shape
 * argument built from {@code tf.shape(x)} means.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/shape">tf.shape</a>
 */
public class Shape extends TensorGenerator {

  /** The parameters of {@code tf.shape(input, out_type=None, name=None)}, after {@code self}. */
  private enum Parameters {
    INPUT,
    OUT_TYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Shape(PointsToSetVariable source) {
    super(source);
  }

  public Shape(CGNode node) {
    super(node);
  }

  /**
   * Resolves the shape vector's shape as {@code (r,)} per resolved operand shape of rank {@code r},
   * {@code (0,)} for a scalar operand, and {@code [Unresolved]} when the operand's rank cannot be
   * read.
   *
   * @param builder The propagation call graph builder.
   * @return The rank-1 shapes; never {@code null} and never empty.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> operandShapes =
        this.shapesOfArg(builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    if (operandShapes == null || operandShapes.isEmpty()) {
      ret.add(List.of(UnresolvedDim.INSTANCE));
      return ret;
    }
    for (List<Dimension<?>> operandShape : operandShapes) {
      // Unknown rank (null) and scalar (empty) are different by contract: only the second has a
      // known extent, zero.
      if (operandShape == null) ret.add(List.of(UnresolvedDim.INSTANCE));
      else ret.add(List.of(new NumericDim(operandShape.size())));
    }
    return ret;
  }

  /**
   * The API's {@code out_type} default, {@code int32}, unless a resolvable {@code out_type} is
   * supplied.
   *
   * @param builder The propagation call graph builder.
   * @return The dtype set; never empty.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.dTypeApiDefaultOrUnknown(builder, Set.of(DType.INT32));
  }

  /**
   * Resolves the shapes of the {@code input} argument: the summary-local points-to set first, then
   * the caller's invoke site. Mirrors {@link Sigmoid}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the argument.
   * @param paramName The keyword parameter name.
   * @return The resolved shapes, or {@code null} if neither path recovers.
   */
  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
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
    return Parameters.OUT_TYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.OUT_TYPE.getName();
  }
}
