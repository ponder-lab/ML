package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.argmax}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/argmax">tf.math.argmax</a>
 */
public class ArgMax extends TensorGenerator {

  public ArgMax(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int inputValNum = this.getArgumentValueNumber(builder, 0, "input", false);
    Set<List<Dimension<?>>> inputShapes = this.getShapes(builder, inputValNum);

    OrdinalSet<InstanceKey> axisPts = this.getArgumentPointsToSet(builder, 1, "axis");
    Set<Integer> axisValues = new HashSet<>();
    if (axisPts != null) {
      for (InstanceKey ik : axisPts) {
        if (ik instanceof ConstantKey) {
          Object val = ((ConstantKey<?>) ik).getValue();
          if (val instanceof Number) axisValues.add(((Number) val).intValue());
        }
      }
    }
    if (axisValues.isEmpty()) axisValues.add(0); // Default axis 0

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      int rank = inputShape.size();
      for (Integer a : axisValues) {
        int normalizedAxis = a < 0 ? a + rank : a;
        if (normalizedAxis >= 0 && normalizedAxis < rank) {
          List<Dimension<?>> newShape = new ArrayList<>(inputShape);
          newShape.remove(normalizedAxis);
          ret.add(newShape);
        }
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.INT64); // argmax returns indices
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
