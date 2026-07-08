package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Base for generators of TensorFlow/NumPy APIs that allocate a tensor from an explicit {@code
 * shape} argument plus an optional {@code dtype} argument (e.g. {@code tf.ones}, {@code tf.zeros},
 * {@code tf.keras.Input}, {@code tf.random.uniform}, {@code tf.one_hot}). It owns the {@code
 * SHAPE}/{@code DTYPE} parameter-slot machinery, the float32 default dtype, the ⊤-shape fallback
 * for an unresolvable shape argument, and the constant-extraction helper. It carries no value
 * semantics, so it is not itself dispatched: concrete subclasses (one per API) extend it.
 * Introduced to replace {@code extends Ones} code-reuse-not-is-a inheritance (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>).
 */
public abstract class TensorTypeAllocator extends TensorGenerator {

  private static final Logger LOGGER = getLogger(TensorTypeAllocator.class.getName());

  protected enum Parameters {
    SHAPE,
    DTYPE,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public TensorTypeAllocator(PointsToSetVariable source) {
    super(source);
  }

  public TensorTypeAllocator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Recovery: an allocator whose dtype argument is another tensor's `.dtype` (e.g.
    // `tf.ones((2, 1), dtype=y.dtype)`) takes its dtype from that tensor. Resolve it rather than
    // taking the float32 default. wala/ML#686.
    Set<DType> fromDTypeAttribute =
        this.getDTypeFromDTypeAttributeArgument(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());
    if (fromDTypeAttribute != null && !fromDTypeAttribute.isEmpty()) {
      LOGGER.fine(
          "Recovered allocator dtype from a `.dtype` argument for source: "
              + describe(source)
              + " -> "
              + fromDTypeAttribute
              + ".");
      return fromDTypeAttribute;
    }

    LOGGER.fine(
        "No dtype specified for source: "
            + describe(source)
            + ". Using default dtype of: "
            + FLOAT32
            + " .");

    // Use the default dtype of float32.
    return EnumSet.of(FLOAT32);
  }

  /**
   * Returns ⊤ (unknown shape) when no shape argument resolves. An allocator like {@code tf.zeros}
   * always produces a tensor, so when the {@code shape} argument can't be resolved statically (e.g.
   * it flows from an unmodeled, content-dependent source such as {@code json.loads(...)}), the
   * correct lattice signal is ⊤ ({@code null}), not ⊥ and not a crash. Recovering such a
   * content-dependent shape is the user-annotation problem tracked by <a
   * href="https://github.com/wala/ML/issues/370">wala/ML#370</a>; this method is only the
   * non-crashing floor beneath it. This previously threw {@link UnsupportedOperationException},
   * which aborted the whole analysis (<a
   * href="https://github.com/wala/ML/issues/604">wala/ML#604</a>). Mirrors the ⊤ signal {@code
   * Input} already used (<a href="https://github.com/wala/ML/issues/355">wala/ML#355</a>) and the
   * lattice conventions in {@code CONTRIBUTING.md}. Logs a {@code FINE} marker at each such site so
   * the ⊤ here stays distinguishable from a genuine ⊤ and the wala/ML#370 annotation worklist is
   * discoverable.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null} (⊤, unknown shape); never an empty set, since the allocation is a tensor.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Recovery: an allocator whose shape argument is another tensor's `.shape` (e.g.
    // `tf.ones(x.shape)`) takes its shape from that tensor. Resolve it rather than dropping to ⊤.
    // wala/ML#604.
    Set<List<Dimension<?>>> fromShapeAttribute =
        this.getShapeFromShapeAttributeArgument(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());
    if (fromShapeAttribute != null && !fromShapeAttribute.isEmpty()) {
      LOGGER.fine(
          "Recovered allocator shape from a `.shape` argument for source: "
              + describe(source)
              + " -> "
              + fromShapeAttribute
              + ".");
      return fromShapeAttribute;
    }

    // Emit a discoverable marker so the ⊤ here is distinguishable from a "true" ⊤ elsewhere: this
    // is exactly the set of sites where a user-supplied shape annotation would help. The set is
    // grep-able as the wala/ML#370 annotation worklist without aborting the rest of the analysis
    // (which is what throwing here did, wala/ML#604).
    LOGGER.fine(
        "Unresolved allocator shape for source: "
            + describe(source)
            + "; returning ⊤ (unknown shape). Candidate for a wala/ML#370 shape annotation.");
    return null;
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

  protected static Optional<Integer> getIntValueFromInstanceKey(InstanceKey instanceKey) {
    if (instanceKey instanceof ConstantKey) {
      ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
      Object value = constantKey.getValue();

      if (value == null) return Optional.empty();
      if (value instanceof Number) return Optional.of(((Number) value).intValue());
      // WALA may model a Python `bool` as a `Boolean` rather than a `Number`; `int(True) == 1` and
      // `int(False) == 0`. Any other non-numeric constant (e.g. a string) has no integer value, so
      // degrade to ⊤ (empty) rather than throwing a `ClassCastException`. wala/ML#590.
      if (value instanceof Boolean) return Optional.of(Boolean.TRUE.equals(value) ? 1 : 0);
      return Optional.empty();
    }

    throw new IllegalArgumentException(
        "Cannot get integer value from non-constant InstanceKey: " + describe(instanceKey) + ".");
  }
}
