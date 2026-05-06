package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.expm1}. Pure passthrough — output shape and dtype both inherit from
 * {@code x}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/expm1">tf.math.expm1</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Expm1 extends PassThroughUnaryTensorGenerator {

  public Expm1(PointsToSetVariable source) {
    super(source);
  }

  public Expm1(CGNode node) {
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
