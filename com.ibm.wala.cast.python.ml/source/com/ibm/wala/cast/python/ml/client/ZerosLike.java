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

  protected enum Parameters {
    INPUT,
    DTYPE,
    NAME,
    LAYOUT,
    SHAPE;

    public String getParameterName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ZerosLike(PointsToSetVariable source) {
    super(source);
  }

  protected int getInputParameterPosition() {
    return Parameters.INPUT.getIndex();
  }

  protected String getInputParameterName() {
    return Parameters.INPUT.getParameterName();
  }

  @Override
  protected int getValueParameterPosition() {
    return this.getInputParameterPosition();
  }

  @Override
  protected String getValueParameterName() {
    return this.getInputParameterName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }
}
