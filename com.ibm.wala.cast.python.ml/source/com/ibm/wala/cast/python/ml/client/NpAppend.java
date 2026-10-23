package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.NpOuter.orUnknown;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.append(arr, values, axis=None)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). Without an {@code axis}, both
 * inputs are flattened and joined, so the result is rank 1 with the sum of their flattened lengths.
 * With one, it is a concatenation along that axis, whose shape this generator does not compute.
 * Either way the dtype is NumPy's promotion of the two inputs'.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/generated/numpy.append.html">numpy.append</a>
 */
public class NpAppend extends NpArrayFunction {

  public NpAppend(PointsToSetVariable source) {
    super(source);
  }

  public NpAppend(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    if (this.isArgumentPassed(builder, 2, "axis")) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> arr : orUnknown(this.getShapesOfArgument(builder, 0)))
      for (List<Dimension<?>> values : orUnknown(this.getShapesOfArgument(builder, 1)))
        ret.add(List.of(combine(size(arr), size(values), true)));
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
}
