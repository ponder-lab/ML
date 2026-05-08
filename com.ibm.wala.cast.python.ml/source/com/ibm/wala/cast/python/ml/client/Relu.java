package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.nn.relu(features, name=None)}. Pure passthrough — output shape and dtype
 * both inherit from {@code features}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/relu">tf.nn.relu</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Relu extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.nn.relu(features, name=None)}. Ordinals
   * match the position in the XML's {@code paramNames} after the implicit {@code self} receiver.
   */
  protected enum Parameters {
    /** Tensor input to the ReLU activation; shape and dtype source for the result. */
    FEATURES,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in arg-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "features"}).
     */
    public String getName() {
      return name().toLowerCase();
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public Relu(PointsToSetVariable source) {
    super(source);
  }

  public Relu(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.FEATURES.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.FEATURES.getName();
  }
}
