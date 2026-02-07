package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
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
 * A generator for {@code tf.layers.dense}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/layers/dense">tf.layers.dense</a>
 */
public class Dense extends TensorGenerator {

  public Dense(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 0, "inputs");
    if (inputPts.isEmpty()) return Collections.emptySet();

    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);

    OrdinalSet<InstanceKey> unitsPts = this.getArgumentPointsToSet(builder, 1, "units");
    Set<Integer> unitsValues = HashSetFactory.make();
    if (unitsPts != null) {
      for (InstanceKey ik : unitsPts) {
        if (ik instanceof ConstantKey) {
          Object val = ((ConstantKey<?>) ik).getValue();
          if (val instanceof Number) unitsValues.add(((Number) val).intValue());
        }
      }
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.isEmpty()) continue;
      for (Integer units : unitsValues) {
        List<Dimension<?>> newShape = new ArrayList<>(inputShape);
        newShape.set(newShape.size() - 1, new NumericDim(units));
        ret.add(newShape);
      }
      if (unitsValues.isEmpty()) {
        List<Dimension<?>> newShape = new ArrayList<>(inputShape);
        newShape.set(newShape.size() - 1, new NumericDim(-1)); // Unknown units
        ret.add(newShape);
      }
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Derive dtype from the input tensor.
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 0, "inputs");
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
