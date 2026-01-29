package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Variable.Parameters.DTYPE;
import static com.ibm.wala.cast.python.ml.client.Variable.Parameters.INITIAL_VALUE;
import static com.ibm.wala.cast.python.ml.client.Variable.Parameters.SHAPE;
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
    SHAPE
  }

  public Variable(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If explicit shape is missing, try inferring from initial_value
    OrdinalSet<InstanceKey> initialValuePts =
        this.getArgumentPointsToSet(
            builder, INITIAL_VALUE.ordinal(), INITIAL_VALUE.name().toLowerCase());

    if (initialValuePts != null && !initialValuePts.isEmpty()) {
      return getShapesOfValue(builder, initialValuePts);
    }

    return emptySet();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If explicit dtype is missing, try inferring from initial_value
    OrdinalSet<InstanceKey> initialValuePts =
        this.getArgumentPointsToSet(
            builder, INITIAL_VALUE.ordinal(), INITIAL_VALUE.name().toLowerCase());

    if (initialValuePts != null && !initialValuePts.isEmpty()) {
      return getDTypesOfValue(builder, initialValuePts);
    }

    return EnumSet.noneOf(DType.class);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // First try explicit shape argument
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(builder, SHAPE.ordinal(), SHAPE.name().toLowerCase());

    if (shapePts != null && !shapePts.isEmpty()) {
      return getShapesFromShapeArgument(builder, shapePts);
    }

    // Fallback to default (infer from initial_value)
    return getDefaultShapes(builder);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    // First try explicit dtype argument
    OrdinalSet<InstanceKey> dtypePts =
        this.getArgumentPointsToSet(builder, DTYPE.ordinal(), DTYPE.name().toLowerCase());

    if (dtypePts != null && !dtypePts.isEmpty()) {
      return getDTypesFromDTypeArgument(builder, dtypePts);
    }

    // Fallback to default (infer from initial_value)
    return getDefaultDTypes(builder);
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE.ordinal();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }
}
