package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.reduce_prod}. The axis-collapse / keepdims shape inference and the
 * dtype-preserving default both come from {@link Reduction}. See wala/ML#449 (Tier 3).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_prod">tf.math.reduce_prod</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReduceProd extends Reduction {

  public ReduceProd(PointsToSetVariable source) {
    super(source);
  }
}
