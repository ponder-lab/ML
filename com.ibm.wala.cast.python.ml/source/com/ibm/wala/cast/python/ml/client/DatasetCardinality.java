package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code Dataset.cardinality()} and {@code tf.data.experimental.cardinality}. Returns
 * a 0-D scalar {@code int64} tensor: the number of elements of the dataset, or {@code
 * tf.data.INFINITE_CARDINALITY} or {@code tf.data.UNKNOWN_CARDINALITY} when it has no finite count
 * the runtime can tell. Both shape and dtype are intrinsic (wala/ML#1072).
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#cardinality">tf.data.Dataset.cardinality</a>
 */
public class DatasetCardinality extends TensorGenerator {

  public DatasetCardinality(PointsToSetVariable source) {
    super(source);
  }

  public DatasetCardinality(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Set.of(Collections.emptyList());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.INT64);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
