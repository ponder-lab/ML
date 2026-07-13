package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * Represents a call to the <code>constant()</code> function in TensorFlow. The value-argument
 * shape/dtype inference lives in {@link ValueExtractingTensorGenerator}; {@code tf.constant} adds
 * an optional explicit {@code shape} argument.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/constant">constant()</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Constant extends ValueExtractingTensorGenerator {

  protected enum Parameters {
    VALUE,
    DTYPE,
    SHAPE,
    NAME,
    VERIFY_SHAPE;

    public String getName() {
      return this.name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return this.ordinal();
    }
  }

  public Constant(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Constant(CGNode node) {
    super(node);
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }
}
