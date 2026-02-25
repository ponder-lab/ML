package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/** A generator for the index tensor produced by {@code tf.data.Dataset.enumerate}. */
public class EnumerateIndexGenerator extends TensorGenerator {

  public EnumerateIndexGenerator(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(Collections.emptyList());
  }

  @Override
  protected int getShapeParameterPosition() {
    return -1;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(DType.INT64);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return -1;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
