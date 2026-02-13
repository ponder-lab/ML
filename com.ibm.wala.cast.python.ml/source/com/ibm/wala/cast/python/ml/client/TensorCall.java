package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.TensorCall.Parameters.DTYPE;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by internal constructors like `tf.Tensor()` or `tf.ndarray()`.
 */
public class TensorCall extends TensorGenerator {

  protected enum Parameters {
    OP,
    VALUE_INDEX,
    DTYPE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public TensorCall(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> opPointsToSet =
        this.getArgumentPointsToSet(builder, Parameters.OP.getIndex(), Parameters.OP.getName());
    if (opPointsToSet.isEmpty()) {
      return Collections.emptySet();
    }
    return this.getShapesOfValue(builder, opPointsToSet);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException(
        "DType is mandatory and must be provided explicitly for tf.Tensor/tf.ndarray.");
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());

    if (pointsToSet == null || pointsToSet.isEmpty()) return this.getDefaultDTypes(builder);

    return this.getDTypesFromDTypeArgument(builder, pointsToSet);
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
    return DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return DTYPE.getName();
  }
}
