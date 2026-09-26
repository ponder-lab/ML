package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None,
 * name=None)}. Pure passthrough: reversing the leading {@code seq_lengths[i]} elements of each
 * batch slice permutes elements within {@code input}, so the output's shape and dtype both inherit
 * from {@code input}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/reverse_sequence">tf.reverse_sequence</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ReverseSequence extends PassThroughUnaryTensorGenerator {

  public ReverseSequence(PointsToSetVariable source) {
    super(source);
  }

  public ReverseSequence(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input";
  }
}
