package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
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
 * A generator for tensors created by the `Input()` function in TensorFlow/Keras.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Input">TensorFlow Input()
 *     API</a>.
 */
public class Input extends TensorTypeAllocator {

  private static final Logger LOGGER = getLogger(Input.class.getName());

  protected enum Parameters {
    SHAPE,
    BATCH_SIZE,
    NAME,
    DTYPE,
    SPARSE,
    TENSOR,
    RAGGED,
    TYPE_SPEC;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Input(PointsToSetVariable source) {
    super(source);
  }

  public Input(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    // `tensor`: the layer wraps an existing tensor, so its dtype passes through verbatim
    // (wala/ML#617). A non-empty points-to set is what signals the argument was actually supplied:
    // in the synthetic `do`-method context every formal parameter has a value number, so the value
    // number alone is not a presence test.
    int tensorValNum =
        this.getArgumentValueNumber(
            builder, this.getTensorParameterPosition(), this.getTensorParameterName(), true);
    if (tensorValNum > 0) {
      OrdinalSet<InstanceKey> tensorPts =
          this.getArgumentPointsToSet(
              builder, this.getTensorParameterPosition(), this.getTensorParameterName());
      if (tensorPts != null && !tensorPts.isEmpty()) return this.getDTypes(builder, tensorValNum);
    }

    // `type_spec`: the dtype comes from the supplied `TypeSpec` object (wala/ML#617).
    int typeSpecValNum =
        this.getArgumentValueNumber(
            builder, this.getTypeSpecParameterPosition(), this.getTypeSpecParameterName(), true);
    if (typeSpecValNum > 0) {
      OrdinalSet<InstanceKey> typeSpecPts =
          this.getArgumentPointsToSet(
              builder, this.getTypeSpecParameterPosition(), this.getTypeSpecParameterName());
      if (typeSpecPts != null && !typeSpecPts.isEmpty())
        return this.getDTypesFromDTypeArgument(builder, typeSpecPts);
    }

    int valNum =
        this.getArgumentValueNumber(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName(), true);
    if (valNum <= 0) return this.getDefaultDTypes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());

    if (pointsToSet == null || pointsToSet.isEmpty())
      // Fallback to default.
      return this.getDefaultDTypes(builder);

    LOGGER.fine(
        "Found possible dtypes: "
            + describe(pointsToSet)
            + " for source: "
            + describe(this.getSource())
            + ".");
    return this.getDTypesFromDTypeArgument(builder, pointsToSet);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // `tensor`: the layer wraps an existing tensor, so its shape passes through verbatim with no
    // batch dimension prepended (wala/ML#617). A non-empty points-to set is what signals the
    // argument was actually supplied: in the synthetic `do`-method context every formal parameter
    // has a value number, so the value number alone is not a presence test.
    int tensorValNum =
        this.getArgumentValueNumber(
            builder, this.getTensorParameterPosition(), this.getTensorParameterName(), true);
    if (tensorValNum > 0) {
      OrdinalSet<InstanceKey> tensorPts =
          this.getArgumentPointsToSet(
              builder, this.getTensorParameterPosition(), this.getTensorParameterName());
      if (tensorPts != null && !tensorPts.isEmpty()) {
        LOGGER.fine(
            "Reading shape from `tensor` argument for source: " + describe(this.getSource()) + ".");
        return this.getShapes(builder, tensorValNum);
      }
    }

    // `type_spec`: the shape comes from the supplied `TypeSpec` object, again with no batch
    // dimension prepended (wala/ML#617).
    int typeSpecValNum =
        this.getArgumentValueNumber(
            builder, this.getTypeSpecParameterPosition(), this.getTypeSpecParameterName(), true);
    if (typeSpecValNum > 0) {
      OrdinalSet<InstanceKey> typeSpecPts =
          this.getArgumentPointsToSet(
              builder, this.getTypeSpecParameterPosition(), this.getTypeSpecParameterName());
      if (typeSpecPts != null && !typeSpecPts.isEmpty())
        return this.getShapesFromShapeArgument(builder, typeSpecPts);
    }

    int shapeValNum =
        this.getArgumentValueNumber(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName(), true);
    Set<List<Dimension<?>>> shapes;

    if (shapeValNum > 0) {
      OrdinalSet<InstanceKey> shapePts =
          this.getArgumentPointsToSet(
              builder, this.getShapeParameterPosition(), this.getShapeParameterName());

      if (shapePts == null || shapePts.isEmpty()) {
        LOGGER.fine(
            "No shapes found for source: "
                + describe(this.getSource())
                + "; using default shapes.");
        shapes = this.getDefaultShapes(builder);
      } else {
        LOGGER.fine(
            "Found possible shape points-to set: "
                + shapePts
                + " for source: "
                + this.getSource()
                + ".");
        shapes = this.getShapesFromShapeArgument(builder, shapePts);
      }
    } else {
      LOGGER.fine(
          "No shapes found for source: " + describe(this.getSource()) + "; using default shapes.");
      shapes = this.getDefaultShapes(builder);
    }

    // Handle `batch_size`.
    int batchSizeValNum =
        this.getArgumentValueNumber(
            builder, this.getBatchSizeParameterPosition(), this.getBatchSizeParameterName(), true);
    Set<Long> batchSizes = new java.util.HashSet<>();

    if (batchSizeValNum > 0) {
      OrdinalSet<InstanceKey> batchSizePts =
          this.getArgumentPointsToSet(
              builder, this.getBatchSizeParameterPosition(), this.getBatchSizeParameterName());

      if (batchSizePts == null || batchSizePts.isEmpty()) {
        LOGGER.fine(
            "Empty points-to set for batch_size argument in source: "
                + this.getSource()
                + "; assuming unknown.");
      } else {
        Set<Long> values = getPossibleLongValues(batchSizePts);
        // A null result means `batch_size` is not statically resolvable (wala/ML#669); leave
        // `batchSizes` empty so the unknown-batch (dynamic-dim) path below applies.
        if (values != null) batchSizes.addAll(values);
      }
    }

    // Treat `null` entries in `batchSizes` (signaling "None" from `getPossibleLongValues`) the
    // same as an empty `batchSizes`: unknown batch dimension, prepend `DynamicDim`.
    // https://github.com/wala/ML/issues/545.
    boolean batchSizeUnknown = batchSizes.isEmpty() || batchSizes.contains(null);
    Set<Long> knownBatchSizes = new java.util.HashSet<>(batchSizes);
    knownBatchSizes.remove(null);
    if (!knownBatchSizes.isEmpty())
      LOGGER.fine(
          "Found possible batch sizes: "
              + knownBatchSizes
              + " for source: "
              + this.getSource()
              + ".");

    // If the base shape is ⊤ (unknown), the prepended batch dim doesn't recover enough info to
    // synthesize a meaningful shape—propagate ⊤ outward. Avoids NPE on the iteration below.
    if (shapes == null) return null;

    Set<List<Dimension<?>>> newShapes = HashSetFactory.make();

    for (List<Dimension<?>> shape : shapes) {
      if (batchSizeUnknown) {
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(DynamicDim.INSTANCE);
        newShape.addAll(shape);
        newShapes.add(newShape);
      }
      for (Long batchSize : knownBatchSizes) {
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(new NumericDim(batchSize.intValue()));
        newShape.addAll(shape);
        newShapes.add(newShape);
      }
    }

    LOGGER.fine("Generated shapes: " + newShapes + " for source: " + describe(source) + ".");
    return newShapes;
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  protected int getBatchSizeParameterPosition() {
    return Parameters.BATCH_SIZE.getIndex();
  }

  protected String getBatchSizeParameterName() {
    return Parameters.BATCH_SIZE.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  protected int getTensorParameterPosition() {
    return Parameters.TENSOR.getIndex();
  }

  protected String getTensorParameterName() {
    return Parameters.TENSOR.getName();
  }

  protected int getTypeSpecParameterPosition() {
    return Parameters.TYPE_SPEC.getIndex();
  }

  protected String getTypeSpecParameterName() {
    return Parameters.TYPE_SPEC.getName();
  }
}
