package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for {@code tf.reduce_sum}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum">tf.math.reduce_sum</a>
 */
public class ReduceSum extends Reduction {

  public ReduceSum(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ReduceSum(CGNode node) {
    super(node);
  }
}
