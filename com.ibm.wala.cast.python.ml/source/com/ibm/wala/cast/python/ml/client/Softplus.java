package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.softplus}. Pure passthrough — output shape and dtype both inherit
 * from {@code features}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/softplus">tf.math.softplus</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Softplus extends PassThroughUnaryTensorGenerator {

  public Softplus(PointsToSetVariable source) {
    super(source);
  }

  public Softplus(CGNode node) {
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
