package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for tensors created by the `zeros_like()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/zeros_like">TensorFlow zeros_like()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ZerosLike extends Constant {

  /**
   * The shape argument is not explicitly provided to zeros_like(); rather, the shape is inferred
   * from the `input` argument.
   */
  private static final int SHAPE_PARAMETER_POSITION = -1;

  public ZerosLike(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }
}
