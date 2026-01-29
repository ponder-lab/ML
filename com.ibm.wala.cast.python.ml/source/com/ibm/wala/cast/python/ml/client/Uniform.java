package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A representation of the `random.uniform()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/random/uniform">TensorFlow uniform()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Uniform extends Ones {

  protected enum Parameters {
    SHAPE,
    MINVAL,
    MAXVAL,
    DTYPE,
    SEED,
    NAME;

    public String getParameterName() {
      return name().toLowerCase();
    }

    public int getParameterIndex() {
      return ordinal();
    }
  }

  public Uniform(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getParameterIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getParameterName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getParameterIndex();
  }

  protected String getShapeParameterName() {
    return Parameters.SHAPE.getParameterName();
  }
}
