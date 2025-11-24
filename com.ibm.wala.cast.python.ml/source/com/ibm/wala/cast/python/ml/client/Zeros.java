package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for tensors created by the `zeros()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/zeros">TensorFlow zeros() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Zeros extends Ones {

  private static final String FUNCTION_NAME = "tf.zeros()";

  public Zeros(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
