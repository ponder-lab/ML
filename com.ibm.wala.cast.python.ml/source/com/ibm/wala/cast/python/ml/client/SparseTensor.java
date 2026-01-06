package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by the `tf.sparse.SparseTensor` constructor.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor">tf.sparse.SparseTensor</a>.
 */
public class SparseTensor extends TensorGenerator {

  private static final int INDICES_PARAMETER_POSITION = 0;
  private static final int VALUES_PARAMETER_POSITION = 1;
  private static final int DENSE_SHAPE_PARAMETER_POSITION = 2;

  public SparseTensor(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Shape is mandatory for SparseTensor (dense_shape).
    // If we are here, it means getShapes() couldn't find the shape argument or it was empty.
    // However, since we return DENSE_SHAPE_PARAMETER_POSITION in getShapeParameterPosition(),
    // getShapes() should have tried to read it.
    // If it's missing, we can't determine the shape.
    throw new IllegalArgumentException(
        "dense_shape argument is mandatory for tf.sparse.SparseTensor");
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // The dtype is inferred from the 'values' argument.
    int valuesValNum = this.getArgumentValueNumber(VALUES_PARAMETER_POSITION);
    return getDTypes(builder, valuesValNum);
  }

  @Override
  protected int getShapeParameterPosition() {
    // TODO: Handle keyword arguments.
    return DENSE_SHAPE_PARAMETER_POSITION;
  }

  @Override
  protected int getDTypeParameterPosition() {
    // No explicit dtype argument. Dtype is inferred from 'values'.
    return -1;
  }
}
