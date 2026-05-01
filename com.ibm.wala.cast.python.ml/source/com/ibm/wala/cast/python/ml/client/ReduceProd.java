package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Set;

/**
 * Generator for {@code tf.math.reduce_prod}. Reuses {@link ReduceMean}'s axis-collapse / keepdims
 * shape inference, but overrides {@link #getDTypes(PropagationCallGraphBuilder)} to inherit the
 * input's dtype directly (no {@code int → float32} promotion that {@link ReduceMean} applies). See
 * wala/ML#449 (Tier 3).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_prod">tf.math.reduce_prod</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReduceProd extends ReduceMean {

  public ReduceProd(PointsToSetVariable source) {
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
