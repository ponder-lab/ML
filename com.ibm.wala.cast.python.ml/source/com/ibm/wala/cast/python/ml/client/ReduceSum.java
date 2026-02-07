package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for {@code tf.reduce_sum}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum">tf.math.reduce_sum</a>
 */
public class ReduceSum extends ReduceMean {

  public ReduceSum(PointsToSetVariable source) {
    super(source);
  }
}
