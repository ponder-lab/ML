package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors produced by iterating a {@code tf.data.TextLineDataset}. Each iteration
 * element is a 0-D scalar string tensor (one line of text per element). See <a
 * href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/TextLineDataset">tf.data.TextLineDataset</a>.
 *
 * <p>Modeled as a distinct {@link DatasetGenerator} subclass with hardcoded element shape ({@code
 * []}, scalar) and dtype ({@link DType#STRING}). The base {@link DatasetGenerator}'s
 * receiver-walking shape/dtype inference can't recover these for {@code TextLineDataset} because
 * the XML model previously aliased it with {@code Ltensorflow/data/Dataset} and there's no upstream
 * tensor-allocation chain to peel — the per-line element type is intrinsic to the API. Same shape
 * as {@link DatasetRangeGenerator}, which does the same trick for {@code tf.data.Dataset.range}
 * (scalars of int64). See wala/ML#452.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TextLineDatasetGenerator extends DatasetGenerator {

  public TextLineDatasetGenerator(PointsToSetVariable source) {
    super(source);
  }

  public TextLineDatasetGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(Collections.emptyList());
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.STRING);
  }
}
