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

  private static final String FUNCTION_NAME = "tf.random.truncated_normal()";

  public TruncatedNormal(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
