package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A generator for the `tf.reshape` operation. It extracts the shape from the `shape` argument,
 * handling `-1` as a symbolic dimension.
 */
public class Reshape extends TensorGenerator {

  public Reshape(PointsToSetVariable source) {
    super(source);
  }

  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Try to get shape from the 'shape' argument (index 1).
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());

    if (shapePts != null && !shapePts.isEmpty()) {
      Set<List<Dimension<?>>> rawShapes = this.getShapesFromShapeArgument(builder, shapePts);
      if (!rawShapes.isEmpty()) {
        Set<List<Dimension<?>>> refinedShapes = HashSetFactory.make();
        for (List<Dimension<?>> shape : rawShapes) {
          List<Dimension<?>> refinedShape = new ArrayList<>();
          for (Dimension<?> dim : shape) {
            if (dim instanceof NumericDim && ((NumericDim) dim).value() == -1) {
              refinedShape.add(new SymbolicDim("?"));
            } else {
              refinedShape.add(dim);
            }
          }
          refinedShapes.add(refinedShape);
        }
        return refinedShapes;
      }
    }

    // 2. Fallback: infer from input tensor (index 0).
    return getDefaultShapes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Infer shape from 'tensor' argument (index 0).
    OrdinalSet<InstanceKey> tensorPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    return this.getShapesOfValue(builder, tensorPts);
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    return this.getDTypesOfValue(builder, tensorPts);
  }

  @Override
  protected int getShapeParameterPosition() {
    return 1;
  }

  @Override
  protected String getShapeParameterName() {
    return "shape";
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  protected int getValueParameterPosition() {
    return 0;
  }

  protected String getValueParameterName() {
    return "tensor";
  }
}
