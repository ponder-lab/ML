package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.exp}. Returns a fresh tensor with the same shape and dtype as the
 * {@code x} input.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/exp">tf.math.exp</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Exp extends PassThroughUnaryTensorGenerator {

  public Exp(PointsToSetVariable source) {
    super(source);
  }

  public Exp(CGNode node) {
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
