package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.cast(x, dtype, name=None)}. Intended output shape inherits from {@code
 * x}; output dtype is the explicit {@code dtype} argument (e.g., {@code tf.int32}, {@code
 * tf.float64}). The base-class dispatch in {@link TensorGenerator#getDTypes} reads the dtype
 * argument when {@link #getDTypeParameterPosition} returns a defined position, so this generator
 * just declares position 1 ({@code dtype}, after {@code x} at 0).
 *
 * <p>The historical {@code pass_through} alias that once won dispatch over this generator (the
 * wala/ML#509 discussion) is gone: {@code tf.cast} is wired to the dedicated summary class, this
 * override is live, {@code testCast} asserts the cast TARGET dtype, and a {@code .dtype} attribute
 * target resolves through the attribute route with an honest unknown terminal (<a
 * href="https://github.com/wala/ML/issues/481">wala/ML#481</a>).
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

  /**
   * A cast preserves its input's SHAPE and transforms its DTYPE, so it declares a {@link
   * TypeFeedKind#SHAPE_ONLY} feed rather than the superclass's pass-through (<a
   * href="https://github.com/wala/ML/issues/481">wala/ML#481</a>). Under a pass-through
   * declaration, a result whose target dtype did not resolve would be written back carrying the
   * INPUT's dtype through the feed channel, reinstating the corruption the default below refuses.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The shape-only feed over the input operand, or {@code null} when none is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.SHAPE_ONLY);
  }

  /**
   * The cast's dtype when the {@code dtype} argument does not resolve through its points-to set.
   * The superclass would pass the INPUT's dtype through, which for a cast asserts the one thing the
   * call exists to change: a bool input then flows on as the result, and a downstream elementwise
   * consumer imposes bool on its float partner (<a
   * href="https://github.com/wala/ML/issues/481">wala/ML#481</a>).
   *
   * <p>The unresolved case is dominated by a {@code .dtype} ATTRIBUTE target ({@code tf.cast(x,
   * y.dtype)}), whose points-to set is always empty because the read allocates nothing, so the
   * target is recovered from the callers' frames first (the wala/ML#686 route, which is where the
   * read lives: this generator anchors on the summary's return value). Failing that, the honest
   * answer is a tensor of unknown dtype, which also breaks the corruption's self-consistency, since
   * an unknown partner makes the downstream coercion decline rather than impose.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The recovered target dtype, or {@code unknown}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> viaAttribute =
        this.getDTypeFromDTypeAttributeArgument(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());
    if (viaAttribute != null && !viaAttribute.isEmpty()) return viaAttribute;
    return EnumSet.of(DType.UNKNOWN);
  }
}
