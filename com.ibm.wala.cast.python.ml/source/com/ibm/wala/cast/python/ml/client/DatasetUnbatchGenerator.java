package com.ibm.wala.cast.python.ml.client;

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
 * replaces only what they do to each shape.
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
   * @implNote Each shape without its leading dimension. A scalar has no rows, and {@code unbatch}
   *     rejects it at run time, so it contributes nothing.
   */
  @Override
  protected Set<List<Dimension<?>>> applyBatching(
      Set<List<Dimension<?>>> inputShapes, PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes)
      if (!shape.isEmpty()) ret.add(new ArrayList<>(shape.subList(1, shape.size())));
    return ret.isEmpty() ? null : ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown. The number of rows is the sum of the receiver's leading dimensions, which
   *     the receiver's size does not give: a batch reports its own receiver's size, not its number
   *     of batches.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
