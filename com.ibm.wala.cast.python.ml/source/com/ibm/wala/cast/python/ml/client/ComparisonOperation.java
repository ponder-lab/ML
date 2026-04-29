package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Element-wise comparison operations in TensorFlow ({@code tf.equal}, {@code tf.not_equal}, {@code
 * tf.less}, {@code tf.less_equal}, {@code tf.greater}, {@code tf.greater_equal}).
 *
 * <p>Differs from {@link ElementWiseOperation} only in dtype: comparison ops always produce a
 * {@code tf.bool}-dtype tensor regardless of input dtypes, per the TensorFlow API contract. Shape
 * inference is the same broadcast behavior inherited from {@link ElementWiseOperation}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/equal">tf.math.equal</a>.
 * @see <a href="https://github.com/wala/ML/issues/427">wala/ML#427</a>.
 */
public class ComparisonOperation extends ElementWiseOperation {

  public ComparisonOperation(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.BOOL);
  }
}
