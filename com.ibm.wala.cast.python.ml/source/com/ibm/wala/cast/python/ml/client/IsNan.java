package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code tf.math.is_nan(x, name=None)}: a {@code bool} tensor of {@code x}'s shape
 * (<a href="https://github.com/wala/ML/issues/1076">wala/ML#1076</a>).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/is_nan">tf.math.is_nan</a>
 */
public class IsNan extends PassThroughUnaryTensorGenerator {

  public IsNan(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public IsNan(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "x";
  }

  /**
   * The predicate preserves its input's shape and gives its own dtype, so it declares a {@link
   * TypeFeedKind#SHAPE_ONLY} feed, as {@link Cast} does, rather than the superclass's pass-through,
   * which would give the result its input's dtype.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The shape-only feed over the input operand, or {@code null} when none is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.SHAPE_ONLY);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.BOOL);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.BOOL);
  }
}
