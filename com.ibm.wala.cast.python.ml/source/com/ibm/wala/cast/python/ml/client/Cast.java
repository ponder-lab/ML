package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.cast}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/cast">tf.cast</a>
 */
public class Cast extends TensorGenerator {

  public Cast(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getShapes(builder, 1); // input is at 1
  }

  @Override
  protected int getDTypeParameterPosition() {
    return 2;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.FLOAT32);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }
}
