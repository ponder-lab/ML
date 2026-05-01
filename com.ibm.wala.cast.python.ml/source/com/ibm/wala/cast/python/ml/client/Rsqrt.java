package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.rsqrt} (reciprocal square root). Returns a fresh tensor with the
 * same shape and dtype as the {@code x} input.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/rsqrt">tf.math.rsqrt</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Rsqrt extends PassThroughUnaryTensorGenerator {

  public Rsqrt(PointsToSetVariable source) {
    super(source);
  }

  public Rsqrt(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "x";
  }
}
