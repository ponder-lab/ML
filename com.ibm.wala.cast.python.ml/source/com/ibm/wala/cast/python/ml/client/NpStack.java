package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code np.stack(arrays, axis=0)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). The shape is {@link Stack}'s:
 * the arrays' common shape with their count inserted at {@code axis}. The dtype is {@link
 * NpConcatenate}'s over the same {@code arrays}, NumPy's promotion across every element, where
 * {@link Stack} reads the first element's as TensorFlow requires one dtype.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.stack.html">numpy.stack</a>
 */
public class NpStack extends Stack {

  public NpStack(PointsToSetVariable source) {
    super(source);
  }

  public NpStack(CGNode node) {
    super(node);
  }

  @Override
  protected String getValuesParameterName() {
    return "arrays";
  }

  /**
   * {@inheritDoc}
   *
   * <p>Read by an {@link NpConcatenate} anchored where this generator is: both functions take their
   * {@code arrays} first, and both promote across them.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    NpConcatenate concatenate =
        this.getSource() != null
            ? new NpConcatenate(this.getSource())
            : new NpConcatenate(this.getNode());
    return concatenate.getDefaultDTypes(builder);
  }

  /**
   * No type feed: {@link Stack}'s feeds the elements' own dtypes, which a promotion replaces.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null}.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return null;
  }

  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
