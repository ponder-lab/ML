package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
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

  public ExpandDims(PointsToSetVariable source) {
    super(source);
  }

  public ExpandDims(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input";
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
