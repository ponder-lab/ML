package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.diag(v, k=0)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). A rank-1 {@code v} of length
 * {@code n} builds a square matrix of side {@code n + |k|}; a rank-2 {@code v} yields its {@code
 * k}-th diagonal, rank 1, whose length this generator computes for the main diagonal only. Any
 * other rank is an error in NumPy and contributes nothing. The dtype is {@code v}'s NumPy dtype.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.diag.html">numpy.diag</a>
 */
public class NpDiag extends NpArrayFunction {

  public NpDiag(PointsToSetVariable source) {
    super(source);
  }

  public NpDiag(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = this.getShapesOfArgument(builder, 0);
    if (shapes == null) return null;
    Integer k =
        this.isArgumentPassed(builder, 1, "k")
            ? this.getConstantIntegerArgument(builder, 1, "k")
            : Integer.valueOf(0);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> v : shapes) {
      if (v.size() == 1) {
        Dimension<?> side =
            k == null
                ? UnresolvedDim.INSTANCE
                : combine(v.get(0), new NumericDim(Math.abs(k)), true);
        ret.add(List.of(side, side));
      } else if (v.size() == 2) {
        ret.add(
            List.of(
                k != null
                        && k == 0
                        && v.get(0) instanceof NumericDim rows
                        && v.get(1) instanceof NumericDim columns
                    ? new NumericDim(Math.min(rows.value(), columns.value()))
                    : UnresolvedDim.INSTANCE));
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.getNumpyDTypesOfArgument(builder, 0);
  }
}
