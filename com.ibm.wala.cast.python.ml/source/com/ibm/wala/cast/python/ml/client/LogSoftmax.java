package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.nn.log_softmax}. Returns a fresh tensor with the same shape and dtype as
 * the {@code logits} input. Note the parameter name {@code logits} (vs. {@code x} for the other
 * Tier-2 unary math ops) — matches the TF API signature.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax">tf.nn.log_softmax</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class LogSoftmax extends PassThroughUnaryTensorGenerator {

  public LogSoftmax(PointsToSetVariable source) {
    super(source);
  }

  public LogSoftmax(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "logits";
  }
}
