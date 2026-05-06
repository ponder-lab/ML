package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.nn.leaky_relu(features, alpha=0.2, name=None)}. Pure passthrough — output
 * shape and dtype both inherit from {@code features}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu">tf.nn.leaky_relu</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class LeakyRelu extends PassThroughUnaryTensorGenerator {

  public LeakyRelu(PointsToSetVariable source) {
    super(source);
  }

  public LeakyRelu(CGNode node) {
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
