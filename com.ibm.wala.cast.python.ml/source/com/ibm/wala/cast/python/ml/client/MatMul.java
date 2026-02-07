package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
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
 * A generator for {@code tf.matmul}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/matmul">tf.matmul</a>
 */
public class MatMul extends TensorGenerator {

  public MatMul(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> aPts = this.getArgumentPointsToSet(builder, 0, "a");
    OrdinalSet<InstanceKey> bPts = this.getArgumentPointsToSet(builder, 1, "b");

    if (aPts.isEmpty() || bPts.isEmpty()) return Collections.emptySet();

    Set<List<Dimension<?>>> aShapes = this.getShapesOfValue(builder, aPts);
    Set<List<Dimension<?>>> bShapes = this.getShapesOfValue(builder, bPts);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (List<Dimension<?>> aShape : aShapes) {
      for (List<Dimension<?>> bShape : bShapes) {
        if (aShape.size() >= 2 && bShape.size() >= 2) {
          List<Dimension<?>> newShape = new ArrayList<>();
          // For simplicity, assuming no transpose and 2D for now.
          // tf.matmul(A, B) where A is [M, K] and B is [K, N] -> [M, N]
          newShape.add(aShape.get(aShape.size() - 2));
          newShape.add(bShape.get(bShape.size() - 1));
          ret.add(newShape);
        }
      }
    }

    if (ret.isEmpty()) {
      return Set.of(Collections.emptyList());
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Derive dtype from the first input argument.
    OrdinalSet<InstanceKey> aPts = this.getArgumentPointsToSet(builder, 0, "a");
    if (aPts.isEmpty()) return EnumSet.noneOf(DType.class);
    return this.getDTypesOfValue(builder, aPts);
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
