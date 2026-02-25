package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A proxy generator for an element of a dataset (e.g., a tuple element) that delegates type
 * inference to the dataset's underlying generator, but is not itself a DatasetGenerator. This
 * ensures that subsequent property reads on this element correctly peel dimensions instead of being
 * treated as dataset iterations.
 */
public class DatasetElementGenerator extends TensorGenerator {

  private final TensorGenerator underlying;

  public DatasetElementGenerator(PointsToSetVariable source, TensorGenerator underlying) {
    super(source);
    this.underlying = underlying;
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return underlying.getTensorTypes(builder);
    }
    return super.getTensorTypes(builder);
  }

  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return underlying.getShapes(builder);
    }
    return super.getShapes(builder);
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return underlying.getDTypes(builder);
    }
    return super.getDTypes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
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
    return Collections.emptySet();
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
