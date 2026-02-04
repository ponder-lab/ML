package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
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
    throw new IllegalArgumentException(
        "Modeling for internal tensor constructor " + this.getSource() + " is missing.");
  }

  @Override
  protected java.util.EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    throw new IllegalArgumentException(
        "DType is mandatory and must be provided explicitly for tf.Tensor/tf.ndarray.");
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.DTYPE.getIndex(), Parameters.DTYPE.getName(), true);
    if (valNum <= 0) return this.getDefaultDTypes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, Parameters.DTYPE.getIndex(), Parameters.DTYPE.getName());

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
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
