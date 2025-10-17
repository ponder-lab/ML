package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for tensors created by the `uniform()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/random/uniform">TensorFlow uniform()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Uniform extends Ones {

  private static final int VALUE_NUMBER_FOR_DTYPE_ARGUMENT = 5;

  public Uniform(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected int getValueNumberForDTypeArgument() {
    return VALUE_NUMBER_FOR_DTYPE_ARGUMENT;
  }
}
