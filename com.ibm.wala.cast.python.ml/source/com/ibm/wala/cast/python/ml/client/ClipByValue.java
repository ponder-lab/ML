package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * Generator for {@code tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)}. Pure
 * passthrough — output shape and dtype both inherit from {@code t} (the tensor being clipped).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/clip_by_value">tf.clip_by_value</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ClipByValue extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.clip_by_value(t, clip_value_min,
   * clip_value_max, name=None)}. Ordinals match the position in the XML's {@code paramNames} after
   * the implicit {@code self} receiver.
   */
  protected enum Parameters {
    /** Tensor being clipped; shape and dtype source for the result. */
    T,

    /** Lower bound for clipping; not consumed by this generator. */
    CLIP_VALUE_MIN,

    /** Upper bound for clipping; not consumed by this generator. */
    CLIP_VALUE_MAX,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in arg-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "t"}).
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
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

  public ClipByValue(PointsToSetVariable source) {
    super(source);
  }

  public ClipByValue(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.T.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.T.getName();
  }
}
