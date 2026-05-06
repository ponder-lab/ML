package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)}. Pure
 * passthrough — output shape and dtype both inherit from {@code t} (the tensor being clipped).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/clip_by_value">tf.clip_by_value</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ClipByValue extends PassThroughUnaryTensorGenerator {

  public ClipByValue(PointsToSetVariable source) {
    super(source);
  }

  public ClipByValue(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "t";
  }
}
