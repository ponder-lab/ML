package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Supplier;
import java.util.logging.Logger;

/**
 * A generator for the `tf.reshape` operation. It extracts the shape from the `shape` argument,
 * handling `-1` as a symbolic dimension.
 */
public class Reshape extends TensorGenerator {

  /** The logger for this class. */
  @SuppressWarnings("unused")
  private static final Logger LOGGER = Logger.getLogger(Reshape.class.getName());

  private enum Parameters {
    /**
     * The input tensor to reshape. This parameter represents the tensor that is being reshaped in
     * the `tf.reshape` operation. It is used to determine the original shape and data type of the
     * tensor,
     */
    TENSOR,

    /**
     * The target shape for the reshape operation. This parameter represents the desired shape of
     * the output tensor after the reshape operation is applied. It can contain dimensions specified
     * as integers, and may include a `-1` to indicate an inferred dimension. The generator will
     * attempt to resolve the `-1` based on the input tensor's shape and the known dimensions in the
     * target shape.
     */
    SHAPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  public Reshape(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Reshape(CGNode node) {
    super(node);
  }

  /**
   * Computes the possible shapes of the reshaped tensor.
   *
   * <p>This method first attempts to retrieve the target shape from the 'shape' argument. If the
   * 'shape' argument contains a {@code -1} dimension (indicating an inferred dimension), it
   * calculates the size of the input tensor and divides it by the product of the known target
   * dimensions to resolve the {@code -1}. If the input tensor shape is not fully known or constant,
   * it falls back to a symbolic dimension {@code ?}.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return A set of possible shapes for the resulting tensor.
   */
  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    return this.getShapeResult(builder).toLegacy();
  }

  /**
   * Record-carrying core of {@link #getShapes(PropagationCallGraphBuilder)} (wala/ML#718): a
   * partially resolvable target shape vector keeps its resolvable members, refined per member, with
   * the unknown remainder riding through to the output.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The resolution result.
   */
  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    ShapeResult target = this.resolveTargetShapes(builder);
    // 2. Fallback: infer from input tensor.
    if (target == null) return ShapeResult.fromLegacy(getDefaultShapes(builder));
    if (target.members().isEmpty()) return target;
    return new ShapeResult(
        refineTargetShapes(target.members(), () -> this.getDefaultShapes(builder)),
        target.hasUnknown());
  }

  /**
   * Resolves the target shapes the {@code shape} argument determines, before any {@code -1}
   * placeholder is inferred from the input. Shared by the substrate arm ({@link #getShapeResult})
   * and the feed's rule ({@link #reshapeRule}), so both read the argument the same way.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The raw target shapes with their unknown remainder; {@link ShapeResult#unknown()} when
   *     the argument is present but determines nothing statically; {@code null} when no shape
   *     argument resolves at all, in which case the caller infers the result from the input.
   */
  private ShapeResult resolveTargetShapes(PropagationCallGraphBuilder builder) {
    // 1. Try to get shape from the 'shape' argument.
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());

    if (shapePts != null && !shapePts.isEmpty()) {
      // `tf.reshape(arr, tf.shape(other))` is a common pattern where the shape argument is itself a
      // runtime Tensor (the result of `tf.shape(...)`); `getShapesFromShapeArgument` degrades such
      // unrecognized forms to ⊤ ("output shape unknown") rather than throwing (wala/ML#471). See
      // wala/ML#538 for the surfacing fixture (`tf2_test_take_along_axis.py`).
      Set<List<Dimension<?>>> rawShapes = this.getShapesFromShapeArgument(builder, shapePts);
      // Soundness: when the `shape` argument is present but unparseable, the output shape is ⊤
      // (the result is determined by `shape`, not the input tensor — falling back to input-shape
      // inference would be unsound).
      if (rawShapes == null) return ShapeResult.unknown();
      if (!rawShapes.isEmpty()) return new ShapeResult(rawShapes, false);
    }

    // The shape argument's points-to set is empty (or held no shape-bearing allocation). A shape
    // vector derived from a tensor's shape (`t.shape.as_list()[-2:]` and friends) has no
    // points-to state at all, so resolve it by def-use provenance instead (wala/ML#703).
    ShapeResult vectorShapes =
        this.getShapeResultFromShapeVectorArgument(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());
    // A partially resolvable target keeps its members; the remainder rides through (wala/ML#718).
    if (!vectorShapes.members().isEmpty()) return vectorShapes;
    // Soundness: a structurally-recognized shape vector whose walk fails (e.g. a bound that
    // isn't statically constant) determines the output shape but is unknown, so the output is ⊤;
    // falling through to input-shape inference would leak the input's shape (wala/ML#704).
    if (this.isShapeVectorArgument(
        builder, this.getShapeParameterPosition(), this.getShapeParameterName()))
      return ShapeResult.unknown();
    return null;
  }

  /**
   * Refines each raw target shape against the input's shapes: a single {@code -1} placeholder is
   * inferred by dividing the input's total size by the product of the known target dimensions, and
   * stays the symbolic {@code ?} when the input is not fully known, the division is not exact, or
   * the target carries more than one placeholder. A target without a placeholder is kept as is.
   *
   * @param rawShapes The target shapes as the {@code shape} argument determines them.
   * @param inputShapes The input's shapes, read only when a placeholder needs them.
   * @return The refined target shapes.
   */
  private static Set<List<Dimension<?>>> refineTargetShapes(
      Set<List<Dimension<?>>> rawShapes, Supplier<Set<List<Dimension<?>>>> inputShapes) {
    Set<List<Dimension<?>>> refinedShapes = HashSetFactory.make();

    for (List<Dimension<?>> shape : rawShapes) {
      int unknownIndex = -1;
      long productKnown = 1;
      boolean canInfer = true;

      // Scan the whole member: breaking out early on a non-numeric dimension skipped the
      // `-1`-placeholder detection for mixed vectors (e.g. an unresolved leading fold alongside
      // a literal `-1`), letting the raw `-1` escape as a fixed size into downstream broadcast
      // and compatibility checks (wala/ML#741).
      for (int i = 0; i < shape.size(); i++) {
        Dimension<?> dim = shape.get(i);
        if (dim instanceof NumericDim) {
          int val = ((NumericDim) dim).value();
          if (val == -1) {
            if (unknownIndex != -1) canInfer = false; // More than one -1
            else unknownIndex = i;
          } else {
            productKnown *= val;
          }
        } else {
          canInfer = false; // Non-numeric dimension; keep scanning for placeholders.
        }
      }

      if (unknownIndex != -1) {
        // We need input shapes to infer -1 dimension.
        Set<List<Dimension<?>>> inputs = inputShapes.get();

        if (canInfer && inputs != null && !inputs.isEmpty()) {
          for (List<Dimension<?>> inputShape : inputs) {
            long inputSize = 1;
            boolean inputKnown = true;
            for (Dimension<?> d : inputShape) {
              if (d instanceof NumericDim) {
                inputSize *= ((NumericDim) d).value();
              } else {
                inputKnown = false;
                break;
              }
            }

            List<Dimension<?>> refinedShape = new ArrayList<>(shape);
            // The -1 dimension is only inferable when the division is exact; a zero known
            // product (any inferred value satisfies 0 * k == 0), a non-exact division, or a
            // quotient outside the non-negative int range leaves it symbolic.
            long inferredDim =
                inputKnown && productKnown != 0 && inputSize % productKnown == 0
                    ? inputSize / productKnown
                    : -1;
            if (inferredDim >= 0 && inferredDim <= Integer.MAX_VALUE) {
              refinedShape.set(unknownIndex, new NumericDim((int) inferredDim));
            } else {
              refinedShape.set(unknownIndex, new SymbolicDim("?"));
            }
            refinedShapes.add(refinedShape);
          }
        } else {
          List<Dimension<?>> refinedShape = new ArrayList<>();
          for (Dimension<?> dim : shape) {
            if (dim instanceof NumericDim && ((NumericDim) dim).value() == -1) {
              refinedShape.add(new SymbolicDim("?"));
            } else {
              refinedShape.add(dim);
            }
          }
          refinedShapes.add(refinedShape);
        }
      } else {
        refinedShapes.add(shape);
      }
    }
    return refinedShapes;
  }

  /**
   * The reshape's shape rule as a function of the input shape alone, its target resolved once from
   * the points-to substrate (the wala/ML#905 form). It is {@link #getShapeResult}'s own computation
   * over one input shape: the target's placeholder is inferred from that shape, a target the
   * substrate cannot determine types nothing, and a missing target falls back to the input's shape
   * exactly as the substrate arm does, so an input typed only by dataflow gets the answer the
   * substrate would have given it.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The rule; it yields {@code null} for a target that is unknown or only partially
   *     resolvable, which the legacy view of the seed already collapses to unknown.
   */
  private ShapeTransform reshapeRule(PropagationCallGraphBuilder builder) {
    ShapeResult target = this.resolveTargetShapes(builder);
    return input -> {
      if (target == null) return Collections.singleton(input);
      if (target.hasUnknown() || target.members().isEmpty()) return null;
      return refineTargetShapes(target.members(), () -> Collections.singleton(input));
    };
  }

  /**
   * Declares a {@link TypeFeedKind#TRANSFORM} feed over the {@code tensor} argument, carrying
   * {@link #reshapeRule} (<a href="https://github.com/wala/ML/issues/940">wala/ML#940</a>). This
   * addresses {@code tf.reshape} alone, not the family of generators that inherit the base's null
   * declaration. Without a feed, a reshape whose input is typed only by the dataflow (a layer-call
   * result, which no generator types at seed time) seeds a member with an unknown dtype that
   * nothing ever removes, while the reshape's dataflow edge adds the same shape with the dtype the
   * input's state delivers: two members of one shape, one unknown, from a single reshape. With the
   * feed, that seed is suppressed and its dtype composes from the input's converged state.
   *
   * <p>The kind is {@link TypeFeedKind#TRANSFORM} rather than {@link TypeFeedKind#DTYPE_ONLY}
   * because the dataflow's reshape node op resolves a {@code -1} placeholder against the members
   * the input edge delivers: a dtype-only feed would strip the input's shape from that edge, and
   * the placeholder would stay symbolic exactly where the node op used to fold it. The rule folds
   * it the same way, so the fed member arrives already reshaped.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The rule-carrying feed over the input's caller-side key, or {@code null} when the input
   *     cannot be located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    CGNode node = this.getNode();
    if (node == null || node.getIR() == null) return null;
    int tensorVn =
        this.getArgumentValueNumber(
            builder, this.getValueParameterPosition(), this.getValueParameterName(), true);
    // A `read_data` anchoring answers with the `Integer.MAX_VALUE` sentinel, which names no local.
    if (tensorVn <= 0 || tensorVn == Integer.MAX_VALUE) return null;
    PointerKey tensor =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, tensorVn);
    return new TypeFeed(TypeFeedKind.TRANSFORM, List.of(tensor), this.reshapeRule(builder));
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Infer shape from the 'tensor' argument: its points-to union first, then the per-context
    // caller walk. A caller-side operator-produced argument (e.g. the matmul-plus-bias feeding the
    // vendored Conv1d's output reshape) has no allocation site, so it is invisible to the
    // argument's points-to set and resolves only through the caller's value-number read
    // (wala/ML#739).
    OrdinalSet<InstanceKey> tensorPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    if (tensorPts != null && !tensorPts.isEmpty()) {
      Set<List<Dimension<?>>> fromValue = this.getShapesOfValue(builder, tensorPts);
      if (fromValue != null && !fromValue.isEmpty()) return fromValue;
    }
    ShapeResult viaCallers =
        this.getArgumentShapeResultViaCallers(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    // The default mode's contract is the resolvable subset (wala/ML#716).
    if (!viaCallers.members().isEmpty()) return viaCallers.members();
    return this.getShapesOfValue(builder, tensorPts);
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    return this.getDTypesOfValue(builder, tensorPts);
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.ordinal();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  protected int getValueParameterPosition() {
    return Parameters.TENSOR.ordinal();
  }

  protected String getValueParameterName() {
    return Parameters.TENSOR.getName();
  }
}
