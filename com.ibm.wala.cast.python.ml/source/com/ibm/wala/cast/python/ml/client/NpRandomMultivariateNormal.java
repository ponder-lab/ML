package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.random.multivariate_normal(mean, cov, size=None)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). Each draw is a vector of the
 * mean's length, so the shape is the {@code size} argument's with that length appended, and a draw
 * without a {@code size} is the vector alone, never a scalar. The dtype is {@code float64}.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html">numpy.random.multivariate_normal</a>
 */
public class NpRandomMultivariateNormal extends NpRandomSizedDraw {

  public NpRandomMultivariateNormal(PointsToSetVariable source) {
    super(source);
  }

  public NpRandomMultivariateNormal(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> sizes =
        super.isScalarDraw(builder) ? Collections.singleton(List.of()) : super.getShapes(builder);
    if (sizes == null || sizes.isEmpty()) return null;
    Set<List<Dimension<?>>> means = this.getArgumentShapesWithFallback(builder, 0, "mean");
    Set<Dimension<?>> lengths = HashSetFactory.make();
    if (means != null)
      for (List<Dimension<?>> mean : means) if (mean.size() == 1) lengths.add(mean.get(0));
    if (lengths.isEmpty()) lengths.add(UnresolvedDim.INSTANCE);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> size : sizes)
      for (Dimension<?> length : lengths) {
        List<Dimension<?>> shape = new ArrayList<>(size);
        shape.add(length);
        ret.add(shape);
      }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Never: a draw without a {@code size} is still a vector.
   */
  @Override
  protected boolean isScalarDraw(PropagationCallGraphBuilder builder) {
    return false;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.FLOAT64);
  }
}
