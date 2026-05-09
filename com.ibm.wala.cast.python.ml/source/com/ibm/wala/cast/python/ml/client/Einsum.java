package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.einsum(equation, *inputs, **kwargs)}. Output shape is left at ⊤ pending
 * equation-string parsing — the precise shape is derivable from the equation's output label
 * sequence and each input's shape, but that requires an einsum-equation parser that this generator
 * doesn't yet implement. Output dtype inherits from the first tensor input (TF promotes per-input
 * dtypes upstream of einsum, so the first input's dtype is the canonical source).
 *
 * <p>Argument layout per {@code tensorflow.xml}:
 *
 * <ul>
 *   <li>position 0: {@code equation} (string).
 *   <li>position 1: first tensor input (varargs); dtype source.
 *   <li>position 2+: additional tensor inputs.
 * </ul>
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/einsum">tf.linalg.einsum</a>
 * @see <a href="https://github.com/wala/ML/issues/449">wala/ML#449</a> (Tier 5).
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Einsum extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.einsum(equation, *inputs, **kwargs)}.
   * Ordinals match the position in {@code tensorflow.xml}'s {@code paramNames} after the implicit
   * {@code self} receiver. Note that {@code equation} is a string (not a tensor), so the dtype
   * source is {@code INPUTS} at position 1.
   */
  protected enum Parameters {
    /** The einsum equation string (e.g. {@code "ij,jk->ik"}); not consumed by this generator. */
    EQUATION,

    /** The first tensor input (the {@code *inputs} varargs); the dtype source. */
    INPUTS;

    /**
     * Lowercase keyword name used in argument-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "inputs"}).
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

  public Einsum(PointsToSetVariable source) {
    super(source);
  }

  public Einsum(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.INPUTS.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.INPUTS.getName();
  }

  /**
   * Override the inherited shape passthrough to ⊤. The first tensor input's shape isn't a sound
   * approximation for einsum's output shape — the equation can permute, contract, or expand dims
   * arbitrarily. Without an einsum-equation parser, ⊤ is the only sound answer. (The dtype
   * passthrough from the parent class IS sound, since einsum preserves the (promoted) dtype.)
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null} — ⊤, unknown shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
