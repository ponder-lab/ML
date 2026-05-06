package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.softsign}. Pure passthrough — output shape and dtype both inherit
 * from {@code features}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/softsign">tf.math.softsign</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Softsign extends PassThroughUnaryTensorGenerator {

  public Softsign(PointsToSetVariable source) {
    super(source);
  }

  public Softsign(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "features";
  }
}
