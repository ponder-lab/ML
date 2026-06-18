package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A generator for tensors created by the `zeros_like()` function in TensorFlow. The value-argument
 * shape/dtype inference lives in {@link ValueExtractingTensorGenerator}; {@code tf.zeros_like}
 * names its value argument {@code input} and has no explicit shape argument.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/zeros_like">TensorFlow zeros_like()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ZerosLike extends ValueExtractingTensorGenerator {

  protected enum Parameters {
    INPUT,
    DTYPE,
    NAME,
    LAYOUT,
    SHAPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
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
    return Parameters.INPUT.getName();
  }

  @Override
  protected int getValueParameterPosition() {
    return this.getInputParameterPosition();
  }

  @Override
  protected String getValueParameterName() {
    return this.getInputParameterName();
  }
}
