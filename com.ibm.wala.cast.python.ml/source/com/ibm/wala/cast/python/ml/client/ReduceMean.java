package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.HashSet;
import java.util.Set;

/**
 * A generator for {@code tf.reduce_mean}. Unlike most reductions, it promotes integer inputs to
 * {@code float32} (the mean of integers is a float), so it overrides the dtype-preserving default
 * in {@link Reduction}. The {@code axis}/{@code keepdims} shape collapse is inherited.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean">tf.math.reduce_mean</a>
 */
public class ReduceMean extends Reduction {

  public ReduceMean(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ReduceMean(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    // reduce_mean promotes integer inputs to float32 (the mean of integers is a float); float
    // inputs
    // keep their dtype. tf.reduce_mean has no dtype argument, so the output is determined by input.
    int inputValNum =
        this.getArgumentValueNumber(
            builder, Parameters.INPUT_TENSOR.getIndex(), Parameters.INPUT_TENSOR.getName(), false);
    Set<DType> inputTypes = this.getDTypes(builder, inputValNum);
    Set<DType> ret = new HashSet<>();
    for (DType t : inputTypes) {
      if (t == DType.INT32 || t == DType.INT64) ret.add(DType.FLOAT32);
      else ret.add(t);
    }
    return ret;
  }
}
