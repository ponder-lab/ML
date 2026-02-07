package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.placeholder}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/placeholder">tf.placeholder</a>
 */
public class Placeholder extends TensorGenerator {

  public Placeholder(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Set.of(Collections.emptyList());
  }

  @Override
  protected int getShapeParameterPosition() {
    return 1;
  }

  @Override
  protected String getShapeParameterName() {
    return "shape";
  }

  @Override
  protected int getDTypeParameterPosition() {
    return 0;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.FLOAT32);
  }
}
