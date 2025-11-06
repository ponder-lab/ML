package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.DEPTH;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.DTYPE;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.OFF_VALUE;
import static com.ibm.wala.cast.python.ml.client.OneHot.Parameters.ON_VALUE;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
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

public class OneHot extends ZerosLike {

  private static final String FUNCTION_NAME = "tf.one_hot()";

  enum Parameters {
    INDICES,
    DEPTH,
    ON_VALUE,
    OFF_VALUE,
    AXIS,
    DTYPE
  }

  public OneHot(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException(
        "Shapes are derived from mandatory numeric arguments and must be provided explicitly.");
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
      if (numArgs == Parameters.DEPTH.ordinal() + 1)
        // Neither on_value nor off_value is provided.
        ret.add(DType.FLOAT32);
      else if (numArgs >= Parameters.ON_VALUE.ordinal() + 1) {
        // Either on_value and off_value are provided. We must at least have the on_value.
        EnumSet<DType> onValueDTypes =
            this.getDTypes(builder, this.getOnValueArgumentValueNumber());

        if (!onValueDTypes.isEmpty()) ret.addAll(onValueDTypes);
        else {
          EnumSet<DType> offValueDTypes =
              this.getDTypes(builder, this.getOffValueArgumentValueNumber());
          ret.addAll(offValueDTypes);
        }
      }

    return ret;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }

  protected int getDepthParameterPosition() {
    return DEPTH.ordinal();
  }

  protected int getOnValueParameterPosition() {
    return ON_VALUE.ordinal();
  }

  protected int getOffValueParameterPosition() {
    return OFF_VALUE.ordinal();
  }

  protected int getOnValueArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getOnValueParameterPosition());
  }

  protected int getOffValueArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getOffValueParameterPosition());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Set<List<Dimension<?>>> indices = this.getShapes(builder, this.getValueArgumentValueNumber());
    int depthArgumentValueNumber = this.getDepthArgumentValueNumber();

    if (depthArgumentValueNumber <= 0)
      throw new IllegalStateException(
          "No depth argument value found for OneHot tensor generation.");

    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    PointerKey pointerKey =
        pointerAnalysis
            .getHeapModel()
            .getPointerKeyForLocal(this.getNode(), depthArgumentValueNumber);
    OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

    if (pointsToSet == null || pointsToSet.isEmpty())
      throw new IllegalStateException(
          "No depth argument value found for OneHot tensor generation.");

    for (InstanceKey instanceKey : pointsToSet) {
      int depth = getIntValueFromInstanceKey(instanceKey);

      // For each shape in indices, append the depth as a new dimension.
      for (List<Dimension<?>> shape : indices) {
        NumericDim dim = new NumericDim(depth);

        List<Dimension<?>> newShape = new ArrayList<>(shape);
        newShape.add(dim);
        ret.add(newShape);
      }
    }

    assert ret.size() >= indices.size()
        : "Number of OneHot shapes should be at least the number of indices shapes.";

    return ret;
  }

  private static int getIntValueFromInstanceKey(InstanceKey instanceKey) {
    if (instanceKey instanceof ConstantKey) {
      ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
      Object value = constantKey.getValue();
      return ((Long) value).intValue();
    }

    throw new IllegalStateException(
        "Cannot get integer value from non-constant InstanceKey: " + instanceKey);
  }

  private int getDepthArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getDepthParameterPosition());
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
