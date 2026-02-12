package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.AXIS;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.DEPTH;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.DTYPE;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.INDICES;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.OFF_VALUE;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.ON_VALUE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

public class OneHot extends Ones {

  private static final int AXIS_END = -1;

  protected enum Parameters {
    INDICES,
    DEPTH,
    ON_VALUE,
    OFF_VALUE,
    AXIS,
    DTYPE;

    public String getName() {
      return name().toLowerCase();
    }
  }

  public OneHot(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If dtype is not provided, it will attempt to assume the data type of on_value or off_value,
    // if one or both are passed in. If none of on_value, off_value, or dtype are provided, dtype
    // will default to the value tf.float32.
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);

    OrdinalSet<InstanceKey> onValPTS =
        this.getArgumentPointsToSet(
            builder, this.getOnValueParameterPosition(), getOnValueParameterName());
    if (onValPTS != null && !onValPTS.isEmpty()) {
      ret.addAll(this.getDTypesOfValue(builder, onValPTS));
    }

    OrdinalSet<InstanceKey> offValPTS =
        this.getArgumentPointsToSet(
            builder, this.getOffValueParameterPosition(), getOffValueParameterName());
    if (offValPTS != null && !offValPTS.isEmpty()) {
      ret.addAll(this.getDTypesOfValue(builder, offValPTS));
    }

    if (ret.isEmpty()) {
      ret.add(FLOAT32);
    }

    return ret;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }

  protected String getDTypeParameterName() {
    return DTYPE.getName();
  }

  protected int getIndicesParameterPosition() {
    return INDICES.ordinal();
  }

  protected String getIndicesParameterName() {
    return INDICES.getName();
  }

  protected int getDepthParameterPosition() {
    return DEPTH.ordinal();
  }

  protected String getDepthParameterName() {
    return DEPTH.getName();
  }

  protected int getAxisParameterPosition() {
    return AXIS.ordinal();
  }

  protected String getAxisParameterName() {
    return AXIS.getName();
  }

  protected int getOnValueParameterPosition() {
    return ON_VALUE.ordinal();
  }

  protected String getOnValueParameterName() {
    return ON_VALUE.getName();
  }

  protected int getOffValueParameterPosition() {
    return OFF_VALUE.ordinal();
  }

  protected String getOffValueParameterName() {
    return OFF_VALUE.getName();
  }

  protected int getOnValueArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getOnValueParameterPosition(), getOnValueParameterName(), true);
  }

  protected int getOffValueArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getOffValueParameterPosition(), getOffValueParameterName(), true);
  }

  protected int getIndicesArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getIndicesParameterPosition(), getIndicesParameterName(), true);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    OrdinalSet<InstanceKey> indicesPTS =
        this.getArgumentPointsToSet(
            builder, this.getIndicesParameterPosition(), getIndicesParameterName());

    if (indicesPTS == null || indicesPTS.isEmpty())
      throw new IllegalStateException(
          "Empty points-to set for "
              + INDICES.name().toLowerCase()
              + " argument in "
              + OneHot.class.getName()
              + ": "
              + this.getNode());

    Set<List<Dimension<?>>> indices = this.getShapesOfValue(builder, indicesPTS);

    OrdinalSet<InstanceKey> depthPTS =
        this.getArgumentPointsToSet(
            builder, this.getDepthParameterPosition(), getDepthParameterName());

    if (depthPTS == null || depthPTS.isEmpty())
      throw new IllegalStateException(
          "No depth argument value found for " + OneHot.class.getName() + " tensor generation.");

    Set<Integer> possibleAxes = this.getPossibleAxes(builder);

    for (int axis : possibleAxes)
      for (InstanceKey depthIK : depthPTS) {
        int depth =
            getIntValueFromInstanceKey(depthIK)
                .orElseThrow(
                    () ->
                        new IllegalStateException(
                            "Depth argument value for "
                                + OneHot.class.getName()
                                + " is not an integer: "
                                + depthIK
                                + "."));

        // For each shape in indices, append the depth as a new dimension.
        for (List<Dimension<?>> shape : indices) {
          NumericDim dim = new NumericDim(depth);
          List<Dimension<?>> newShape = new ArrayList<>(shape);

          if (axis == AXIS_END) newShape.add(dim);
          else newShape.add(axis, dim);

          ret.add(newShape);
        }
      }

    assert ret.size() >= indices.size()
        : "Number of "
            + OneHot.class.getName()
            + " shapes should be at least the number of "
            + INDICES.name().toLowerCase()
            + " shapes.";

    return ret;
  }

  private Set<Integer> getPossibleAxes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, this.getAxisParameterPosition(), this.getAxisParameterName(), true);
    if (valNum <= 0) return Set.of(AXIS_END);

    Set<Integer> ret = HashSetFactory.make();

    OrdinalSet<InstanceKey> axisPTS =
        this.getArgumentPointsToSet(
            builder, this.getAxisParameterPosition(), this.getAxisParameterName());

    if (axisPTS == null || axisPTS.isEmpty())
      // Fallback to default.
      return ret;

    for (InstanceKey instanceKey : axisPTS)
      ret.add(getIntValueFromInstanceKey(instanceKey).orElse(AXIS_END));

    return ret;
  }
}
