package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code tf.math.reduce_all}. Reuses {@link ReduceMean}'s axis-collapse / keepdims
 * shape inference; output dtype is always {@link DType#BOOL} (the runtime API requires a bool input
 * and returns bool). See wala/ML#449 (Tier 3).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_all">tf.math.reduce_all</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReduceAll extends Reduction {

  public ReduceAll(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.BOOL);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.BOOL);
  }
}
