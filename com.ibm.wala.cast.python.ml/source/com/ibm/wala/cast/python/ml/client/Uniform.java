package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A representation of the `random.uniform()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/random/uniform">TensorFlow uniform()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Uniform extends Ones {

  private static final String FUNCTION_NAME = "tf.random.uniform()";

  private static final int DTYPE_PARAMETER_POSITION = 3;

  public Uniform(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
