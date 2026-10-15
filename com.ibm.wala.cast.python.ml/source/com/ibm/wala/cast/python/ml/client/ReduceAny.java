package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.reduce_any}, also spelled {@code tf.reduce_any}. It types its result
 * as {@link ReduceAll} does: {@link Reduction}'s axis and {@code keepdims} handling for the shape,
 * and {@code bool} for the dtype, since both reduce a {@code bool} tensor to a {@code bool} tensor
 * and differ only in the logical operation (<a
 * href="https://github.com/wala/ML/issues/1076">wala/ML#1076</a>).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_any">tf.math.reduce_any</a>
 */
public class ReduceAny extends ReduceAll {

  public ReduceAny(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ReduceAny(CGNode node) {
    super(node);
  }
}
