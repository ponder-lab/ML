package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Set;

/**
 * A generator for tensors created by the `tf.sparse.SparseTensor` constructor.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor">tf.sparse.SparseTensor</a>.
 */
public class SparseTensor extends Ones {

  protected enum Parameters {
    INDICES,
    VALUES,
    DENSE_SHAPE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseTensor(PointsToSetVariable source) {
    super(source);
  }

  /** The dtype is inferred from the 'values' argument. */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int valuesValNum =
        this.getArgumentValueNumber(
            builder, this.getValuesParameterPosition(), this.getValuesParameterName(), false);
    return getDTypes(builder, valuesValNum);
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
