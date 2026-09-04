package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A generator for tensors created by the `ones_like()` function in TensorFlow. The value-argument
 * shape/dtype inference lives in {@link ValueExtractingTensorGenerator}; {@code tf.ones_like} names
 * its value argument {@code input} and has no explicit shape argument.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/ones_like">TensorFlow ones_like()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class OnesLike extends ValueExtractingTensorGenerator {

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

  public OnesLike(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public OnesLike(CGNode node) {
    super(node);
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
