package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/** A tensor generator for the `tf.reshape()` function. */
public class Reshape extends TensorGenerator {

  public Reshape(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected int getShapeParameterPosition() {
    return 1;
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If we can't determine the dtype from arguments (reshape usually preserves dtype),
    // we try to inspect the first argument (the tensor being reshaped).
    // The first argument is at position 0.
    int tensorArgValueNumber = getArgumentValueNumber(0);
    if (tensorArgValueNumber > 0) {
      return getDTypes(builder, tensorArgValueNumber);
    }
    return EnumSet.noneOf(DType.class);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Attempt to determine the shape from the second argument (the shape argument).
    // The shape argument is at position 1.
    return getShapes(builder, getArgumentValueNumber(1));
  }
}
