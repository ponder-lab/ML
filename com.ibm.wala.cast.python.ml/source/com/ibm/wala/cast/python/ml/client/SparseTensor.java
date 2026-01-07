package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.util.Util.getFunction;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import java.util.EnumSet;

/**
 * A generator for tensors created by the `tf.sparse.SparseTensor` constructor.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor">tf.sparse.SparseTensor</a>.
 */
public class SparseTensor extends Ones {

  /**
   * The SparseTensor constructor does not have an explicit 'dtype' argument. The dtype is inferred
   * from the 'values' argument.
   */
  private static final int DTYPE_PARAMETER_POSITION = -1;

  @SuppressWarnings("unused")
  private static final int INDICES_PARAMETER_POSITION = 0;

  private static final int VALUES_PARAMETER_POSITION = 1;

  private static final int DENSE_SHAPE_PARAMETER_POSITION = 2;

  public SparseTensor(PointsToSetVariable source) {
    super(source);
  }

  /** The dtype is inferred from the 'values' argument. */
  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // TODO: Handle keyword arguments.
    int valuesValNum = this.getArgumentValueNumber(VALUES_PARAMETER_POSITION);
    return getDTypes(builder, valuesValNum);
  }

  @Override
  protected int getShapeParameterPosition() {
    // TODO: Handle keyword arguments.
    return DENSE_SHAPE_PARAMETER_POSITION;
  }

  /** No explicit dtype argument. Dtype is inferred from 'values'. */
  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }

  @Override
  protected String getSignature() {
    TypeReference function = getFunction(this.getSource());
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }
}
