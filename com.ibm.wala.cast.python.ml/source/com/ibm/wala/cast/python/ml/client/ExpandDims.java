package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.expand_dims(input, axis, name=None)}. Output dtype inherits from {@code
 * input}; output shape is {@code input.shape} with a length-1 dim inserted at {@code axis}.
 * Currently emits ⊤ shape (the precise insertion-at-axis composition is left for a follow-up).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/expand_dims">tf.expand_dims</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ExpandDims extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.expand_dims(input, axis, name=None)}.
   * Ordinals match the position in the XML's {@code paramNames} after the implicit {@code self}
   * receiver.
   */
  protected enum Parameters {
    /** Tensor whose rank is being expanded; shape and dtype source. */
    INPUT,

    /** Position at which to insert the new length-1 dimension; not consumed by this generator. */
    AXIS,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in arg-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "input"}).
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

  public ExpandDims(PointsToSetVariable source) {
    super(source);
  }

  public ExpandDims(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.INPUT.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.INPUT.getName();
  }

  /**
   * Override the inherited shape passthrough to ⊤ — {@code expand_dims} inserts a new length-1 dim
   * at {@code axis}, so the input's shape is not the output's shape (rank differs by 1). Composing
   * the precise output shape requires reading the input's shape and the constant {@code axis};
   * deferred to follow-up. Dtype passthrough from the parent class IS sound.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null} — ⊤, unknown shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
