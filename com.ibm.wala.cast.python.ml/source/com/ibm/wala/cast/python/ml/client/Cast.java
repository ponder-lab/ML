package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * Generator for {@code tf.cast(x, dtype, name=None)}. Output shape inherits from {@code x}; output
 * dtype is the explicit {@code dtype} argument (e.g., {@code tf.int32}, {@code tf.float64}). The
 * base-class dispatch in {@link TensorGenerator#getDTypes} reads the dtype argument when {@link
 * #getDTypeParameterPosition} returns a defined position, so this generator just declares position
 * 1 ({@code dtype}, after {@code x} at 0). Shape inheritance is provided by the {@link
 * PassThroughUnaryTensorGenerator} base, which reads the input arg's shape via {@link
 * #getInputParameterPosition} / {@link #getInputParameterName}.
 *
 * <p>The {@code tf.cast} {@code pass_through} alias in {@code tensorflow.xml} previously won
 * dispatch over this generator, erasing the dtype change. An empirical probe in <a
 * href="https://github.com/wala/ML/issues/499">wala/ML#499</a> measured the alias-removal blast
 * radius and confirmed that the chained-consumer regression which <a
 * href="https://github.com/wala/ML/issues/509">wala/ML#509</a> was meant to guard against does not
 * surface in the current test suite; the alias has been removed.
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
