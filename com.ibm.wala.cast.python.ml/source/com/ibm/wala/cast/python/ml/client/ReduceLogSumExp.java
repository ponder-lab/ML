package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Set;

/**
 * Generator for {@code tf.math.reduce_logsumexp}. Reuses {@link ReduceMean}'s axis-collapse /
 * keepdims shape inference, but overrides {@link #getDTypes(PropagationCallGraphBuilder)} to
 * inherit the input's dtype directly. {@code reduce_logsumexp} requires a float input at runtime,
 * but static analysis just preserves whatever the input dtype was. See wala/ML#449 (Tier 3).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp">tf.math.reduce_logsumexp</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReduceLogSumExp extends ReduceMean {

  public ReduceLogSumExp(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int inputValNum =
        this.getArgumentValueNumber(
            builder, Parameters.INPUT_TENSOR.getIndex(), Parameters.INPUT_TENSOR.getName(), false);
    return this.getDTypes(builder, inputValNum);
  }
}
