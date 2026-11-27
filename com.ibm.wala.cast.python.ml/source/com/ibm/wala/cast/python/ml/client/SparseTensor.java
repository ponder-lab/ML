package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for tensors created by the `tf.sparse.SparseTensor` constructor.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor">tf.sparse.SparseTensor</a>.
 */
public class SparseTensor extends TensorTypeAllocator {

  @Override
  protected boolean producesSparseTensor() {
    return true;
  }

  protected enum Parameters {
    INDICES,
    VALUES,
    DENSE_SHAPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  protected SparseTensor(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  protected SparseTensor(CGNode node) {
    super(node);
  }

  /**
   * The dtype is inferred from the 'values' argument. The constructor yields a sparse tensor
   * whatever it is given, so a {@code values} the analysis cannot read, such as an attribute of an
   * unmodeled library object, contributes the unknown dtype rather than none, which would erase the
   * tensor (wala/ML#1344).
   *
   * <p>An empty read that is not final is the exception, following {@code
   * ElementWiseOperation.resultDTypes} (wala/ML#1094): a loop-carried {@code values} is read while
   * its own cycle is still converging, and unknown there would be a member its later dtypes cannot
   * displace, since the dtype join absorbs into unknown. The result is ⊥ instead, and {@link
   * #isObservationFinal} schedules the query to be recomputed against the settled state, where a
   * read that is still empty gives unknown.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int valuesValNum =
        this.getArgumentValueNumber(
            builder, this.getValuesParameterPosition(), this.getValuesParameterName(), false);
    Set<DType> ret = this.getDTypes(builder, valuesValNum);
    if (ret != null && !ret.isEmpty()) return ret;
    if (ret != null && !isObservationFinal(builder)) return EnumSet.noneOf(DType.class);
    return EnumSet.of(DType.UNKNOWN);
  }

  protected int getIndicesParameterPosition() {
    return Parameters.INDICES.getIndex();
  }

  protected String getIndicesParameterName() {
    return Parameters.INDICES.getName();
  }

  protected int getValuesParameterPosition() {
    return Parameters.VALUES.getIndex();
  }

  protected String getValuesParameterName() {
    return Parameters.VALUES.getName();
  }

  protected int getDenseShapeParameterPosition() {
    return Parameters.DENSE_SHAPE.getIndex();
  }

  protected String getDenseShapeParameterName() {
    return Parameters.DENSE_SHAPE.getName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return this.getDenseShapeParameterPosition();
  }

  @Override
  protected String getShapeParameterName() {
    return this.getDenseShapeParameterName();
  }

  /** No explicit dtype argument. Dtype is inferred from 'values'. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
