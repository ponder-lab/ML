package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.outer(a, b)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). Both inputs are flattened, so
 * the result is rank 2 whatever their ranks, with the flattened length of {@code a} and of {@code
 * b} as its extents, and its dtype is NumPy's promotion of theirs.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.outer.html">numpy.outer</a>
 */
public class NpOuter extends NpArrayFunction {

  public NpOuter(PointsToSetVariable source) {
    super(source);
  }

  public NpOuter(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> a : orUnknown(this.getShapesOfArgument(builder, 0)))
      for (List<Dimension<?>> b : orUnknown(this.getShapesOfArgument(builder, 1)))
        ret.add(List.of(size(a), size(b)));
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.getPromotedDTypesOfArguments(builder, 0, 1);
  }

  @Override
  protected int[] getDTypePositions() {
    return new int[] {0, 1};
  }

  /**
   * The given shapes, or a single unknown shape when there are none, for a read whose result keeps
   * its rank whatever its input's.
   *
   * @param shapes The shapes, or {@code null}.
   * @return The shapes, or a singleton holding {@code null}.
   */
  static Set<List<Dimension<?>>> orUnknown(Set<List<Dimension<?>>> shapes) {
    return shapes == null || shapes.isEmpty() ? Collections.singleton(null) : shapes;
  }
}
