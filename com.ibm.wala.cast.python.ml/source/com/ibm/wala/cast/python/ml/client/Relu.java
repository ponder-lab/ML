package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.nn.relu(features, name=None)}. Pure passthrough — output shape and dtype
 * both inherit from {@code features}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/relu">tf.nn.relu</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Relu extends PassThroughUnaryTensorGenerator {

  public Relu(PointsToSetVariable source) {
    super(source);
  }

  public Relu(CGNode node) {
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
