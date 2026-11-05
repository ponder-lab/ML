package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.unbatch} (wala/ML#1214). The element
 * is a row of the receiver's, component by component: each shape loses its leading dimension. It
 * reads the receiver's element through the batch family's whole, per-index and per-path reads, and
 * replaces only what they do to each shape. A component that can only be a scalar has no rows, and
 * {@code unbatch} rejects it at run time, so it is no tensor.
 */
public class DatasetUnbatchGenerator extends DatasetBatchGenerator {

  public DatasetUnbatchGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetUnbatchGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Each shape without its leading dimension. A scalar has no rows, so it contributes
   *     nothing, and only scalars give no shape at all.
   */
  @Override
  protected Set<List<Dimension<?>>> applyBatching(
      Set<List<Dimension<?>>> inputShapes, PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes)
      if (!shape.isEmpty()) ret.add(new ArrayList<>(shape.subList(1, shape.size())));
    return ret;
  }

  /**
   * The dtypes paired with given shapes: none where the shapes are none, as where only a scalar is
   * unbatched.
   *
   * @param shapes The shapes read for the same component.
   * @param dTypes The receiver's dtypes for it.
   * @return The dtypes, or none.
   */
  private static Set<DType> pairedDTypes(Set<List<Dimension<?>>> shapes, Set<DType> dTypes) {
    return shapes != null && shapes.isEmpty() ? Collections.emptySet() : dTypes;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return pairedDTypes(this.getDefaultShapes(builder), super.getDefaultDTypes(builder));
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return pairedDTypes(
        this.getShapesForIndex(builder, index), super.getDTypesForIndex(builder, index));
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    return pairedDTypes(
        this.getShapesForPath(builder, path), super.getDTypesForPath(builder, path));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown. The number of rows is the sum of the receiver's leading dimensions, which
   *     the receiver's size does not give.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
