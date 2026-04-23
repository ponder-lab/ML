package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Modeling of the function-style {@code numpy.array(x, dtype)} call. Preserves the shape of the
 * first positional argument ({@code x}) and applies the second positional argument as the output
 * dtype, mirroring {@link AstypeOperation}'s shape-preserving / dtype-changing semantics for the
 * method-style {@code x.astype(dtype)} counterpart.
 */
public class NpArray extends TensorGenerator {
  private static final Logger LOGGER = Logger.getLogger(NpArray.class.getName());

  public NpArray(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int sourceVn = getArgumentValueNumber(0);
    LOGGER.fine(() -> "NpArray.getDefaultShapes: source=" + source + ", sourceVn=" + sourceVn);
    if (sourceVn > 0) {
      try {
        Set<List<Dimension<?>>> shapes = getShapes(builder, getNode(), sourceVn);
        LOGGER.fine(
            () -> "NpArray.getDefaultShapes: shapes from sourceVn=" + sourceVn + " -> " + shapes);
        if (shapes != null && !shapes.isEmpty()) {
          return shapes;
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "NpArray.getDefaultShapes: source shape lookup failed for sourceVn=" + sourceVn,
            e);
      }
    }
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int dtypeVn = getArgumentValueNumber(1);
    if (dtypeVn > 0) {
      Set<DType> dTypes = getDTypes(builder, dtypeVn);
      if (!dTypes.isEmpty()) {
        return dTypes;
      }
    }
    return Set.of(DType.UNKNOWN);
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
    return 1;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }
}
