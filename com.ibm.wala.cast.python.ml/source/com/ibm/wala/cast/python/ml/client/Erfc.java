package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.erfc}. Pure passthrough — output shape and dtype both inherit from
 * {@code x}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/erfc">tf.math.erfc</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Erfc extends PassThroughUnaryTensorGenerator {

  public Erfc(PointsToSetVariable source) {
    super(source);
  }

  public Erfc(CGNode node) {
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
