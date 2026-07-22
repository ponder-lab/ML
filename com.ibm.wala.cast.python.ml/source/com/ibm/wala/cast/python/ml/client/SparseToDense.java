package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for the dense tensor produced by {@code tf.sparse.to_dense}. The dense result has the
 * same shape and dtype as the {@code sp_input} SparseTensor operand (the operand's {@code
 * dense_shape} and {@code values} dtype), so they are read from the operand's value number via the
 * recursive {@link #getShapes}/{@link #getDTypes} resolution — which, unlike a direct points-to-set
 * read, is unaffected by the operand's frequently-empty PTS. Unlike its operand, the result is
 * dense.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/sparse/to_dense">tf.sparse.to_dense</a>.
 */
public class SparseToDense extends TensorTypeAllocator {

  protected enum Parameters {
    SP_INPUT,
    DEFAULT_VALUE,
    VALIDATE_INDICES,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseToDense(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public SparseToDense(CGNode node) {
    super(node);
  }

  /** The dense result's shape is the {@code sp_input} SparseTensor's (its {@code dense_shape}). */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int spValueNumber =
        this.getArgumentValueNumber(
            builder, Parameters.SP_INPUT.getIndex(), Parameters.SP_INPUT.getName(), false);
    return this.getShapes(builder, spValueNumber);
  }

  /** The dense result's dtype is the {@code sp_input} SparseTensor's (its {@code values} dtype). */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int spValueNumber =
        this.getArgumentValueNumber(
            builder, Parameters.SP_INPUT.getIndex(), Parameters.SP_INPUT.getName(), false);
    return this.getDTypes(builder, spValueNumber);
  }

  /** No explicit shape argument; the shape comes from the {@code sp_input} operand. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /** No explicit dtype argument; the dtype comes from the {@code sp_input} operand. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
