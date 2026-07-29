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
 * A generator for tensors produced by iterating a {@code tf.data.Dataset.list_files} dataset. Each
 * iteration element is a 0-D scalar string tensor (one matched filename per element), typically
 * consumed by a {@code map(load_*)} callback whose parameter therefore types as {@code {[]
 * string}}. See <a
 * href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#list_files">tf.data.Dataset.list_files</a>.
 *
 * <p>Same intrinsic-element-type trick as {@link TextLineDatasetGenerator} (wala/ML#452), keyed on
 * the distinct {@code ListFilesDataset} allocation the summary makes (the wala/ML#618 treatment).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ListFilesDatasetGenerator extends DatasetGenerator {

  public ListFilesDatasetGenerator(PointsToSetVariable source) {
    super(source);
  }

  public ListFilesDatasetGenerator(CGNode node) {
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
