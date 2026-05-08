package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.reduce_min}. Mirrors {@link ReduceMax} — same axis-collapse /
 * keepdims shape semantics inherited from {@link ReduceMean}, with input dtype passthrough (no
 * {@code int → float32} promotion).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_min">tf.math.reduce_min</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReduceMin extends ReduceMax {

  public ReduceMin(PointsToSetVariable source) {
    super(source);
  }
}
