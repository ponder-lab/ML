package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.nn.max_pool}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/max_pool">tf.nn.max_pool</a>
 */
public class MaxPool extends TensorGenerator {

  public MaxPool(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 0, "value");
    if (inputPts.isEmpty()) return Collections.emptySet();
    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.size() == 4) { // [batch, h, w, c]
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(inputShape.get(0));
        for (int i = 1; i < 3; i++) {
          Dimension<?> d = inputShape.get(i);
          if (d instanceof NumericDim) {
            newShape.add(new NumericDim(((NumericDim) d).value() / 2));
          } else {
            newShape.add(d);
          }
        }
        newShape.add(inputShape.get(3));
        ret.add(newShape);
      }
    }

    if (ret.isEmpty()) {
      return Set.of(Collections.emptyList());
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 0, "value");
    if (inputPts.isEmpty()) return EnumSet.noneOf(DType.class);
    return this.getDTypesOfValue(builder, inputPts);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
