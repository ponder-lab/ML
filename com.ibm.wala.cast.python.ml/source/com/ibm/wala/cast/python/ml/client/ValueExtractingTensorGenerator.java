package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Base for generators of TensorFlow APIs that produce a tensor whose shape and dtype are inferred
 * from a value/input argument (e.g. {@code tf.constant}, {@code tf.zeros_like}, {@code
 * tf.convert_to_tensor}). It owns the value-argument shape/dtype inference, with the value at
 * position 0 and the dtype at position 1 by default; there is no explicit shape argument by
 * default. Subclasses override the value accessor when their argument is named differently (e.g.
 * {@code zeros_like}'s {@code input}) and add an explicit shape argument when the API has one (e.g.
 * {@code constant}). Replaces {@code extends Constant} code-reuse-not-is-a inheritance (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>).
 */
public abstract class ValueExtractingTensorGenerator extends TensorGenerator {

  protected enum Parameters {
    VALUE,
    DTYPE,
    NAME;

    public String getName() {
      return this.name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return this.ordinal();
    }
  }

  public ValueExtractingTensorGenerator(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ValueExtractingTensorGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If the shape argument is not specified, then the shape is inferred from the shape of value.
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pointsToSet);
    // An empty result means the shape could not be inferred — treat as unknown (null / ⊤).
    // `getShapesOfValue` documents that it returns `null` (not an empty set) when the
    // value's points-to set is empty; honor that contract instead of NPE-ing.
    return (shapes == null || shapes.isEmpty()) ? null : shapes;
  }

  /**
   * {@inheritDoc}
   *
   * <p>If the <code>dtype</code> argument is not specified, then the type is inferred from the type
   * of value.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    Set<DType> dTypes = this.getDTypesOfValue(builder, pointsToSet);
    // An empty result means the dtype could not be inferred — treat as unknown
    // (⊤). The produced value is always a tensor; we just cannot pin down its dtype.
    // `getDTypesOfValue` documents that it returns `null` (not an empty set) when
    // the value's points-to set is empty; honor that contract instead of NPE-ing.
    return (dTypes == null || dTypes.isEmpty()) ? EnumSet.of(DType.UNKNOWN) : dTypes;
  }

  protected int getValueArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getValueParameterPosition());
  }

  protected int getValueParameterPosition() {
    return Parameters.VALUE.getIndex();
  }

  protected String getValueParameterName() {
    return Parameters.VALUE.getName();
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
