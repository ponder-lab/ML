package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * Generator for {@code tf.cast(x, dtype, name=None)}. Intended output shape inherits from {@code
 * x}; output dtype is the explicit {@code dtype} argument (e.g., {@code tf.int32}, {@code
 * tf.float64}). The base-class dispatch in {@link TensorGenerator#getDTypes} reads the dtype
 * argument when {@link #getDTypeParameterPosition} returns a defined position, so this generator
 * just declares position 1 ({@code dtype}, after {@code x} at 0).
 *
 * <p><b>Current limitation</b>: in the as-shipped state, this {@code getDTypes} override is
 * bypassed because the {@code tf.cast} {@code pass_through} alias in {@code tensorflow.xml} wins
 * dispatch (the alias is intentionally retained &mdash; removing it breaks downstream tensor flow
 * for chained consumers like {@code reshape(cast(...))}; see <a
 * href="https://github.com/wala/ML/issues/509">wala/ML#509</a>). The analyzer therefore reports the
 * input dtype rather than the cast target, and {@code testCast} asserts the input dtype. Tracked by
 * <a href="https://github.com/wala/ML/issues/481">wala/ML#481</a>.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/cast">tf.cast</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Cast extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.cast(x, dtype, name=None)}. Ordinals match
   * the position in the XML's {@code paramNames} after the implicit {@code self} receiver.
   */
  protected enum Parameters {
    /** Tensor whose elements are cast; shape source. */
    X,

    /** Target dtype for the cast (e.g., {@code tf.int32}); dtype source. */
    DTYPE,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in arg-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "x"}).
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

  public Cast(PointsToSetVariable source) {
    super(source);
  }

  public Cast(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.X.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.X.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
