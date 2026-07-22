package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.reduce_max}. The axis-collapse / keepdims shape inference and the
 * dtype-preserving default both come from {@link Reduction}. See wala/ML#449 (Tier 3).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_max">tf.math.reduce_max</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReduceMax extends Reduction {

  public ReduceMax(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ReduceMax(CGNode node) {
    super(node);
  }
}
