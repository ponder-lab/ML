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
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class OneHot extends TensorTypeAllocator {

  private static final int AXIS_END = -1;

  protected enum Parameters {
    INDICES,
    DEPTH,
    ON_VALUE,
    OFF_VALUE,
    AXIS,
    DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  protected OneHot(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  protected OneHot(CGNode node) {
    super(node);
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

    if (indicesPTS == null || indicesPTS.isEmpty()) {
      return Collections.emptySet();
    }

    Set<List<Dimension<?>>> indices = this.getShapesOfValue(builder, indicesPTS);
    // The output is the indices' shape plus the depth axis, so an unknown indices shape (null,
    // wala/ML#978) leaves the output shape unknown.
    if (indices == null) return null;

    OrdinalSet<InstanceKey> depthPTS =
        this.getArgumentPointsToSet(
            builder, this.getDepthParameterPosition(), getDepthParameterName());

    // A depth the analysis cannot read is a fixed extent it could not compute (wala/ML#1022), as a
    // depth whose only member is None is below.
    if (depthPTS == null) depthPTS = OrdinalSet.empty();

    // The depth axis per `depth` member. TensorFlow raises on a None depth, so a None member is a
    // value no execution reaching the call holds (an unbound default overwritten before the call)
    // and contributes no axis. A member that is not an integer is a fixed extent the analysis
    // could not compute, `Unresolved` by the wala/ML#721 criterion; so is a depth whose only member
    // is None, since the overwrite the analysis missed is then the only way the call runs.
    Set<Dimension<?>> depthDims = HashSetFactory.make();
    for (InstanceKey depthIK : depthPTS) {
      if (depthIK instanceof ConstantKey<?> constant && constant.getValue() == null) continue;
      depthDims.add(
          getIntValueFromInstanceKey(depthIK)
              .<Dimension<?>>map(NumericDim::new)
              .orElse(UnresolvedDim.INSTANCE));
    }
    if (depthDims.isEmpty()) depthDims.add(UnresolvedDim.INSTANCE);

    // `null` when the axis may be any position (wala/ML#1022).
    Set<Integer> possibleAxes = this.getPossibleAxes(builder);

    for (Dimension<?> dim : depthDims)
      // For each shape in indices, insert the depth as a new dimension.
      for (List<Dimension<?>> shape : indices) {
        Set<Integer> axes = possibleAxes;
        if (axes == null) {
          axes = HashSetFactory.make();
          for (int axis = 0; axis < shape.size(); axis++) axes.add(axis);
          axes.add(AXIS_END);
        }
        for (int axis : axes) {
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
            + INDICES.getName()
            + " shapes.";

    return ret;
  }

  /**
   * The positions the depth axis may take: the {@code axis} argument's values, the last when it is
   * omitted, or any, when it is supplied with a value the analysis cannot read (wala/ML#1022).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The positions, {@link #AXIS_END} for the last, or {@code null} for any position.
   */
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
      // An omitted axis is the last; one supplied with a value the analysis cannot read may be any
      // position (wala/ML#1022).
      return Boolean.FALSE.equals(
              this.isArgumentSyntacticallySupplied(
                  builder, this.getAxisParameterPosition(), this.getAxisParameterName()))
          ? Set.of(AXIS_END)
          : null;

    for (InstanceKey instanceKey : axisPTS)
      ret.add(getIntValueFromInstanceKey(instanceKey).orElse(AXIS_END));

    return ret;
  }
}
