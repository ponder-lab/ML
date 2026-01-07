package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A representation of the `tf.random.truncated_normal' API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal">tf.random.truncated_normal
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TruncatedNormal extends Normal {

  public TruncatedNormal(PointsToSetVariable source) {
    super(source);
  }
}
