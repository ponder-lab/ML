package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.Variable` call in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/Variable">tf.Variable</a>.
 */
public class Variable extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Variable.class.getName());

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

  /**
   * Constructs a manual-node Variable generator for {@link TensorGenerator#createManualGenerator}
   * dispatch. Used when the analyzer reaches a {@code Ltensorflow/python/ops/variables/Variable}
   * allocation inside the {@code Variable.do()} summary without a caller-side {@link
   * PointsToSetVariable}; the caller-walk in {@link #getDefaultShapes}/{@link #getDefaultDTypes}
   * then recovers the initial-value shape/dtype from the script's {@code tf.Variable(...)} invoke.
   *
   * @param node The {@link com.ibm.wala.ipa.callgraph.CGNode} for {@code Variable.do()}.
   */
  public Variable(com.ibm.wala.ipa.callgraph.CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If explicit shape is missing, try inferring from initial_value
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName(), true);
    if (valNum <= 0) return null;

    OrdinalSet<InstanceKey> initialValuePts =
        this.getArgumentPointsToSet(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName());

    if (initialValuePts == null || initialValuePts.isEmpty())
      // Shape cannot be determined from initial_value.
      return null;

    // If the chain from `initialValuePts` resolves to concrete shapes, use them. Otherwise return
    // `null` (⊤) — matching the empty-PTS path, so downstream stays tensor-typed at ⊤ rather than
    // dropping to ⊥. Needed because `getShapesOfValue` returns empty (not `null`) when the
    // value's allocation is in a `__call__` summary that `getShapesFromTensor`'s do-only branch
    // can't trace (wala/ML#407).
    Set<List<Dimension<?>>> result = this.getShapesOfValue(builder, initialValuePts);
    if (result != null && !result.isEmpty()) return result;

    // `getShapesOfValue` couldn't resolve (typically because the value's allocation is in a
    // `__call__` summary that `getShapesFromTensor`'s do-only branch can't trace). Walk callers
    // directly for the `initial_value` arg's shape.
    Set<List<Dimension<?>>> viaCallers =
        this.getArgumentShapesViaCallers(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName());
    if (viaCallers != null && !viaCallers.isEmpty()) {
      LOGGER.fine(() -> "Recovered initial_value shapes via caller walk: " + viaCallers);
      return viaCallers;
    }

    LOGGER.fine(
        () ->
            "Empty shapes from initial_value PTS (size="
                + initialValuePts.size()
                + ") and caller walk; falling back to ⊤.");
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If explicit dtype is missing, try inferring from initial_value
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName(), true);
    if (valNum <= 0) return EnumSet.of(DType.UNKNOWN);

    OrdinalSet<InstanceKey> initialValuePts =
        this.getArgumentPointsToSet(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName());

    if (initialValuePts == null || initialValuePts.isEmpty())
      // Dtype cannot be determined.
      return EnumSet.of(DType.UNKNOWN);

    // If the chain from `initialValuePts` resolves to a concrete dtype, use it. Otherwise fall
    // back to `{UNKNOWN}` — matching the empty-PTS path, so downstream stays tensor-typed at ⊤
    // rather than dropping to ⊥. Needed because `getDTypesOfValue` returns empty (not
    // `{UNKNOWN}`) when the value's allocation is in a `__call__` summary that
    // `getDTypesFromTensor`'s do-only branch can't trace (wala/ML#407).
    Set<DType> result = this.getDTypesOfValue(builder, initialValuePts);
    if (result != null && !result.isEmpty()) return result;

    Set<DType> viaCallers =
        this.getArgumentDTypesViaCallers(
            builder, Parameters.INITIAL_VALUE.getIndex(), Parameters.INITIAL_VALUE.getName());
    if (viaCallers != null && !viaCallers.isEmpty()) {
      LOGGER.fine(() -> "Recovered initial_value dtypes via caller walk: " + viaCallers);
      return viaCallers;
    }

    LOGGER.fine(
        () ->
            "Empty dtypes from initial_value PTS (size="
                + initialValuePts.size()
                + ") and caller walk; falling back to {UNKNOWN}.");
    return EnumSet.of(DType.UNKNOWN);
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
