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
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
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
    // TODO: Handle keyword arguments.
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    Set<Integer> possiblePositionalArguments = this.getNumberOfPossiblePositionalArguments(builder);

    for (int numArgs : possiblePositionalArguments)
      if (numArgs == DEPTH.ordinal() + 1)
        // Neither on_value nor off_value is provided. Default to float32.
        ret.add(FLOAT32);
      else if (numArgs == ON_VALUE.ordinal() + 1) {
        // Only on_value may be provided.
        ret.addAll(this.getDTypes(builder, this.getOnValueArgumentValueNumber(builder)));

        // If on_value has no known dtypes, default to float32.
        if (ret.isEmpty()) ret.add(FLOAT32);
      } else if (numArgs >= ON_VALUE.ordinal() + 1) {
        // Either on_value and off_value may be provided.
        ret.addAll(this.getDTypes(builder, this.getOnValueArgumentValueNumber(builder)));
        ret.addAll(this.getDTypes(builder, this.getOffValueArgumentValueNumber(builder)));

        // If neither on_value nor off_value have known dtypes, default to float32.
        if (ret.isEmpty()) ret.add(FLOAT32);
      } else
        throw new IllegalStateException(
            "Unexpected number of positional arguments: " + numArgs + ".");

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
    Set<List<Dimension<?>>> indices =
        this.getShapes(builder, this.getIndicesArgumentValueNumber(builder));
    int depthArgumentValueNumber = this.getDepthArgumentValueNumber(builder);

    if (depthArgumentValueNumber <= 0)
      throw new IllegalStateException(
          "No depth argument value found for OneHot tensor generation.");

    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    PointerKey depthPointerKey =
        pointerAnalysis
            .getHeapModel()
            .getPointerKeyForLocal(this.getNode(), depthArgumentValueNumber);

    OrdinalSet<InstanceKey> depthPTS = pointerAnalysis.getPointsToSet(depthPointerKey);

    if (depthPTS == null || depthPTS.isEmpty())
      throw new IllegalStateException(
          "No depth argument value found for OneHot tensor generation.");

    Set<Integer> possibleAxes = this.getPossibleAxes(builder);

    for (int axis : possibleAxes)
      for (InstanceKey depthIK : depthPTS) {
        int depth =
            getIntValueFromInstanceKey(depthIK)
                .orElseThrow(
                    () ->
                        new IllegalStateException(
                            "Depth argument value for OneHot is not an integer: " + depthIK + "."));

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
        : "Number of OneHot shapes should be at least the number of indices shapes.";

    return ret;
  }

  private Set<Integer> getPossibleAxes(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    Set<Integer> ret = HashSetFactory.make();

    // TODO: Handle keyword arguments.
    for (int numArgs : this.getNumberOfPossiblePositionalArguments(builder)) {
      if (numArgs <= Parameters.AXIS.ordinal())
        // Axis argument not provided.
        ret.add(AXIS_END);
      else { // Axis argument may be provided.
        int axisArgumentValueNumber = this.getAxisArgumentValueNumber(builder);

        PointerKey pointerKey =
            pointerAnalysis
                .getHeapModel()
                .getPointerKeyForLocal(this.getNode(), axisArgumentValueNumber);

        OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

        if (pointsToSet == null || pointsToSet.isEmpty())
          // No axis argument value found; default to AXIS_END.
          ret.add(AXIS_END);
        else
          for (InstanceKey instanceKey : pointsToSet)
            ret.add(getIntValueFromInstanceKey(instanceKey).orElse(AXIS_END));
      }
    }

    return ret;
  }

  private int getDepthArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getDepthParameterPosition(), getDepthParameterName(), true);
  }

  private int getAxisArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getAxisParameterPosition(), getAxisParameterName(), true);
  }
}
