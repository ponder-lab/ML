package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
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
   * Argument-position constant for the dtype source. Position 0 is {@code equation} (a string, not
   * a tensor); position 1 is the first tensor input.
   */
  private static final int FIRST_TENSOR_INPUT_POSITION = 1;

  /** Keyword name for the first tensor input (the {@code *inputs} varargs). */
  private static final String FIRST_TENSOR_INPUT_NAME = "inputs";

  public Einsum(PointsToSetVariable source) {
    super(source);
  }

  public Einsum(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return FIRST_TENSOR_INPUT_POSITION;
  }

  @Override
  protected String getInputParameterName() {
    return FIRST_TENSOR_INPUT_NAME;
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
