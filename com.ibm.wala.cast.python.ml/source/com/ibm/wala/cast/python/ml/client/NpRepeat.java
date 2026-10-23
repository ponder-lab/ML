package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.NpOuter.orUnknown;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.repeat(a, repeats, axis=None)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). Without an {@code axis}, the
 * input is flattened and each element repeated, so the result is rank 1, {@code repeats} times the
 * flattened length when {@code repeats} is an integer constant. With an {@code axis}, the input's
 * rank is kept, which this generator does not compute. The dtype is the input's NumPy dtype, so a
 * Python {@code int} repeated is {@code int64}.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/generated/numpy.repeat.html">numpy.repeat</a>
 */
public class NpRepeat extends NpArrayFunction {

  public NpRepeat(PointsToSetVariable source) {
    super(source);
  }

  public NpRepeat(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    if (this.isArgumentPassed(builder, 2, "axis")) return null;
    Integer repeats = this.getConstantIntegerArgument(builder, 1, "repeats");
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> a : orUnknown(this.getShapesOfArgument(builder, 0))) {
      Dimension<?> size = size(a);
      ret.add(
          List.of(
              repeats != null && repeats >= 0
                  ? combine(size, new NumericDim(repeats), false)
                  : size instanceof DynamicDim ? size : UnresolvedDim.INSTANCE));
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.getNumpyDTypesOfArgument(builder, 0);
  }
}
