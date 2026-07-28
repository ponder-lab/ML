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
 * A generator for tensors produced by iterating a {@code tf.data.TFRecordDataset}. Each iteration
 * element is a 0-D scalar string tensor (one serialized record per element), typically consumed by
 * a {@code map(parse_*)} callback whose parameter therefore types as {@code {[] string}}. See <a
 * href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/TFRecordDataset">tf.data.TFRecordDataset</a>.
 *
 * <p>Same intrinsic-element-type trick as {@link TextLineDatasetGenerator} (wala/ML#452): the
 * per-record element type is a property of the API, with no upstream tensor-allocation chain to
 * peel.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TFRecordDatasetGenerator extends DatasetGenerator {

  public TFRecordDatasetGenerator(PointsToSetVariable source) {
    super(source);
  }

  public TFRecordDatasetGenerator(CGNode node) {
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
