package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.vstack(tup)} (wala/ML#1159): the arrays joined along their first axis
 * after each array of rank below 2 is raised to a row, a 1-D array of shape {@code (n,)} to {@code
 * (1, n)} and a scalar to {@code (1, 1)}. The dtype is NumPy's promotion over the arrays, as for
 * {@link NpConcatenate}.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/generated/numpy.vstack.html">numpy.vstack</a>
 */
public class NpVstack extends NpConcatenate {

  public NpVstack(PointsToSetVariable source) {
    super(source);
  }

  public NpVstack(CGNode node) {
    super(node);
  }

  @Override
  protected String getValuesParameterName() {
    return "tup";
  }

  /**
   * Always {@code 0}: {@code np.vstack} has no {@code axis} parameter.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code 0}.
   */
  @Override
  protected Integer resolveConstantAxis(PropagationCallGraphBuilder builder) {
    return 0;
  }

  @Override
  protected Set<List<Dimension<?>>> promoteElementShapes(Set<List<Dimension<?>>> shapes) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : shapes) {
      if (shape.size() >= 2) {
        ret.add(shape);
        continue;
      }
      List<Dimension<?>> row = new ArrayList<>();
      row.add(new NumericDim(1));
      row.add(shape.isEmpty() ? new NumericDim(1) : shape.get(0));
      ret.add(row);
    }
    return ret;
  }
}
