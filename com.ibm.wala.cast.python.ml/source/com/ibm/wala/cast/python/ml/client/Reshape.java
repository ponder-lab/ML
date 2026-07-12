package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
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
    // 1. Try to get shape from the 'shape' argument.
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());

    boolean hasUnknown = false;
    Set<List<Dimension<?>>> rawShapes = null;

    if (shapePts != null && !shapePts.isEmpty()) {
      // `tf.reshape(arr, tf.shape(other))` is a common pattern where the shape argument is itself a
      // runtime Tensor (the result of `tf.shape(...)`); `getShapesFromShapeArgument` degrades such
      // unrecognized forms to ⊤ ("output shape unknown") rather than throwing (wala/ML#471). See
      // wala/ML#538 for the surfacing fixture (`tf2_test_take_along_axis.py`).
      rawShapes = this.getShapesFromShapeArgument(builder, shapePts);
      // Soundness: when the `shape` argument is present but unparseable, the output shape is ⊤
      // (the result is determined by `shape`, not the input tensor — falling back to input-shape
      // inference would be unsound).
      if (rawShapes == null) return ShapeResult.unknown();
    }

    if (rawShapes == null || rawShapes.isEmpty()) {
      // The shape argument's points-to set is empty (or held no shape-bearing allocation). A shape
      // vector derived from a tensor's shape (`t.shape.as_list()[-2:]` and friends) has no
      // points-to state at all, so resolve it by def-use provenance instead (wala/ML#703).
      ShapeResult vectorShapes =
          this.getShapeResultFromShapeVectorArgument(
              builder, this.getShapeParameterPosition(), this.getShapeParameterName());
      if (!vectorShapes.members().isEmpty()) {
        // A partially resolvable target keeps its members; the remainder rides through
        // (wala/ML#718).
        rawShapes = vectorShapes.members();
        hasUnknown = vectorShapes.hasUnknown();
      }
      // Soundness: a structurally-recognized shape vector whose walk fails (e.g. a bound that
      // isn't statically constant) determines the output shape but is unknown, so the output is ⊤;
      // falling through to input-shape inference would leak the input's shape (wala/ML#704).
      else if (this.isShapeVectorArgument(
          builder, this.getShapeParameterPosition(), this.getShapeParameterName()))
        return ShapeResult.unknown();
    }

    if (rawShapes != null && !rawShapes.isEmpty()) {
      Set<List<Dimension<?>>> refinedShapes = HashSetFactory.make();

      for (List<Dimension<?>> shape : rawShapes) {
        int unknownIndex = -1;
        long productKnown = 1;
        boolean canInfer = true;

        for (int i = 0; i < shape.size(); i++) {
          Dimension<?> dim = shape.get(i);
          if (dim instanceof NumericDim) {
            int val = ((NumericDim) dim).value();
            if (val == -1) {
              if (unknownIndex != -1) {
                canInfer = false; // More than one -1
                break;
              }
              unknownIndex = i;
            } else {
              productKnown *= val;
            }
          } else {
            canInfer = false; // Non-numeric dimension
            break;
          }
        }

        if (unknownIndex != -1) {
          // We need input shapes to infer -1 dimension.
          Set<List<Dimension<?>>> inputShapes = this.getDefaultShapes(builder);

          if (canInfer && inputShapes != null && !inputShapes.isEmpty()) {
            for (List<Dimension<?>> inputShape : inputShapes) {
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
      return new ShapeResult(refinedShapes, hasUnknown);
    }

    // 2. Fallback: infer from input tensor.
    return ShapeResult.fromLegacy(getDefaultShapes(builder));
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Infer shape from 'tensor' argument.
    OrdinalSet<InstanceKey> tensorPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
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
