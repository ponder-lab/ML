package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A representation of the `tf.Variable` call in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/Variable">tf.Variable</a>.
 */
public class Variable extends TensorGenerator {

  protected enum Parameters {
    INITIAL_VALUE,
    TRAINABLE,
    VALIDATE_SHAPE,
    CACHING_DEVICE,
    NAME,
    VARIABLE_DEF,
    DTYPE,
    IMPORT_SCOPE,
    CONSTRAINT,
    SYNCHRONIZATION,
    AGGREGATION,
    SHAPE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Variable(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If explicit shape is missing, try inferring from initial_value
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName(), true);
    if (valNum <= 0) return emptySet();

    OrdinalSet<InstanceKey> initialValuePts =
        this.getArgumentPointsToSet(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName());

    if (initialValuePts == null || initialValuePts.isEmpty())
      // Fallback to default (empty).
      return emptySet();

    return this.getShapesOfValue(builder, initialValuePts);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If explicit dtype is missing, try inferring from initial_value
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName(), true);
    if (valNum <= 0) return EnumSet.noneOf(DType.class);

    OrdinalSet<InstanceKey> initialValuePts =
        this.getArgumentPointsToSet(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName());

    if (initialValuePts == null || initialValuePts.isEmpty())
      // Fallback to default (empty).
      return EnumSet.noneOf(DType.class);

    return this.getDTypesOfValue(builder, initialValuePts);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // First try explicit shape argument
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.SHAPE.getIndex(), Parameters.SHAPE.getName(), true);
    if (valNum > 0) {
      OrdinalSet<InstanceKey> shapePts =
          this.getArgumentPointsToSet(
              builder, Parameters.SHAPE.getIndex(), Parameters.SHAPE.getName());

      if (shapePts == null || shapePts.isEmpty())
        // Fallback to default.
        return this.getDefaultShapes(builder);

      return this.getShapesFromShapeArgument(builder, shapePts);
    }

    // Fallback to default (infer from initial_value)
    return this.getDefaultShapes(builder);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    // First try explicit dtype argument
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.DTYPE.getIndex(), Parameters.DTYPE.getName(), true);
    if (valNum > 0) {
      OrdinalSet<InstanceKey> dtypePts =
          this.getArgumentPointsToSet(
              builder, Parameters.DTYPE.getIndex(), Parameters.DTYPE.getName());

      if (dtypePts == null || dtypePts.isEmpty())
        // Fallback to default.
        return this.getDefaultDTypes(builder);

      return this.getDTypesFromDTypeArgument(builder, dtypePts);
    }

    // Fallback to default (infer from initial_value)
    return this.getDefaultDTypes(builder);
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
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
