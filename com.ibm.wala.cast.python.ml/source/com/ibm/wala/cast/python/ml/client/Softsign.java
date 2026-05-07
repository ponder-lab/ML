package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.softsign}. Pure passthrough — output shape and dtype both inherit
 * from {@code features}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/softsign">tf.math.softsign</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Softsign extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.math.softsign(features, name=None)}.
   * Ordinals match the position in {@code tensorflow.xml}'s {@code paramNames} after the implicit
   * {@code self} receiver, so {@code Parameters.FEATURES.getIndex() == 0} resolves to the first
   * user-facing positional argument.
   */
  protected enum Parameters {
    /** The input tensor; shape and dtype source. */
    FEATURES,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
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

  public Softsign(PointsToSetVariable source) {
    super(source);
  }

  public Softsign(CGNode node) {
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
