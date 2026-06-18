package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.math.argmax}. Output dtype defaults to {@link DType#INT64} (the TF
 * default); when the call passes an explicit {@code output_type} argument (e.g. {@code
 * output_type=tf.int32}), the canonical dtype-arg dispatch from {@link TensorGenerator#getDTypes}
 * is inlined here (in this class's {@link #getDTypes} override; see its Javadoc) to bypass {@link
 * Reduction#getDTypes}, which would otherwise inherit the input tensor's dtype &mdash; the wrong
 * answer for argmax, which always returns an integer index. The override resolves the {@code
 * output_type} argument via {@link #getDTypeParameterPosition} / {@link #getDTypeParameterName} and
 * uses it instead of the default &mdash; fix for <a
 * href="https://github.com/wala/ML/issues/463">wala/ML#463</a>. Output shape is the input shape
 * with the {@code axis} dimension removed (delegated to {@link Reduction}'s keepdims=false
 * reduction). Earlier this was left at ⊤ because the precise shape regressed {@code
 * testNeuralNetwork*}: the per-context shape union (e.g. {@code [256]} train vs. {@code [10000]}
 * test) cross-products through {@code ElementWiseOperation}'s strict broadcast check. That
 * regression was resolved by per-context layer-output allocations under configurable k-CFA (<a
 * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>, <a
 * href="https://github.com/wala/ML/issues/379">wala/ML#379</a>), so the precise shape is now
 * emitted. See <a href="https://github.com/wala/ML/issues/449">wala/ML#449</a> (Tier 6).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/argmax">tf.math.argmax</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Argmax extends Reduction {

  /**
   * Parameter positions and keyword names for {@code tf.math.argmax(input, axis=None,
   * output_type=tf.dtypes.int64, name=None)}. Ordinals match the position in {@code
   * tensorflow.xml}'s {@code paramNames} after the implicit {@code self} receiver, so {@code
   * Parameters.INPUT.getIndex() == 0} resolves to the first user-facing positional argument. The
   * trailing {@code DIMENSION} entry preserves back-compat with the deprecated TF 1.x alias for
   * {@code axis}.
   */
  protected enum Parameters {
    /** The tensor to scan; the index of its max element along {@code axis} is returned. */
    INPUT,

    /** The axis along which to scan; defaults to {@code None}. */
    AXIS,

    /** Optional dtype override for the index output (defaults to {@code int64}). */
    OUTPUT_TYPE,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME,

    /** TF 1.x alias for {@code axis}; retained for back-compat. */
    DIMENSION;

    /**
     * Lowercase keyword name used in argument-resolution helpers when the call site uses {@code
     * keyword=value} syntax. Uses {@link Locale#ROOT} so the conversion is locale-stable (a
     * Turkish-locale JVM lowercasing {@code DIMENSION} would otherwise produce {@code dımensıon}
     * with dotless {@code ı}, breaking keyword lookup for {@code dimension}).
     *
     * @return The lowercased enum name (e.g. {@code "output_type"}).
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

  public Argmax(PointsToSetVariable source) {
    super(source);
  }

  /**
   * {@code argmax}'s input parameter is named {@code input} (not {@code reduce_mean}'s {@code
   * input_tensor}), so resolving it by the superclass name would fail keyword calls like {@code
   * tf.math.argmax(input=x, axis=0)}.
   *
   * @return The positional index of {@code argmax}'s {@code input} parameter.
   */
  @Override
  protected int getInputTensorParameterPosition() {
    return Parameters.INPUT.getIndex();
  }

  /**
   * {@code argmax}'s input parameter is named {@code input} (not {@code reduce_mean}'s {@code
   * input_tensor}), so resolving it by the superclass name would fail keyword calls like {@code
   * tf.math.argmax(input=x, axis=0)}.
   *
   * @return The keyword name ({@code input}) of {@code argmax}'s input parameter.
   */
  @Override
  protected String getInputTensorParameterName() {
    return Parameters.INPUT.getName();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // argmax removes the `axis` dimension (no keepdims), which is exactly Reduction's
    // keepdims=False reduction. With per-context layer-output allocations (wala/ML#530), the input
    // shape is resolved per caller, so the result no longer collapses across contexts.
    //
    // Reduction resolves the reduction axis from the `axis` parameter only, so argmax's deprecated
    // TF 1.x `dimension` alias is not honored for shape (a `tf.argmax(x, dimension=0)` call would
    // reduce as `axis=None`). Honoring the alias precisely is tracked by wala/ML#572.
    return super.getDefaultShapes(builder);
  }

  /**
   * Forces {@code keepdims=false}: {@code argmax} (and {@code argmin}) have no {@code keepdims}
   * parameter and always remove the scanned axis. This also prevents {@link Reduction} from
   * misreading {@code argmax}'s {@code output_type} argument &mdash; which sits at the same
   * positional index as {@code Reduction}'s {@code keepdims} &mdash; as a {@code keepdims} flag,
   * which would otherwise union a spurious {@code keepdims=true} shape for a positional call like
   * {@code tf.argmax(x, 0, tf.int32)}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code Set.of(false)}.
   */
  @Override
  protected Set<Boolean> getKeepDimsValues(PropagationCallGraphBuilder builder) {
    return Set.of(false);
  }

  /**
   * The default output dtype when no {@code output_type} argument is passed. {@link
   * TensorGenerator#getDTypes} dispatches here only when the dtype-arg path returns no resolved
   * dtype, so this is the fallback. TF 2.9 default is {@code int64}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code EnumSet.of(DType.INT64)}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.INT64);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.OUTPUT_TYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.OUTPUT_TYPE.getName();
  }

  /**
   * Bypasses {@link Reduction#getDTypes} (which inherits from the input tensor's dtype) by inlining
   * the canonical {@link TensorGenerator#getDTypes} dispatch: read the {@code output_type} argument
   * if present, otherwise fall back to {@link #getDefaultDTypes}. {@code argmax} produces an
   * integer index regardless of input dtype, so the input-dtype path is never the right answer
   * here.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The output dtype: the resolved {@code output_type} when explicitly passed, or {@code
   *     int64} (the TF default) otherwise.
   */
  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName(), true);
    if (valNum <= 0) return this.getDefaultDTypes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());
    if (pointsToSet == null || pointsToSet.isEmpty()) return this.getDefaultDTypes(builder);
    return this.getDTypesFromDTypeArgument(builder, pointsToSet);
  }
}
