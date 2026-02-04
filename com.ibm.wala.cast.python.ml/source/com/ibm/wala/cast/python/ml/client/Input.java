package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by the `Input()` function in TensorFlow/Keras.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Input">TensorFlow Input()
 *     API</a>.
 */
public class Input extends Ones {

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
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Input(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
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

    LOGGER.info("Found possible dtypes: " + pointsToSet + " for source: " + this.getSource() + ".");
    return this.getDTypesFromDTypeArgument(builder, pointsToSet);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    this.checkUnimplementedParameters(builder);

    int shapeValNum =
        this.getArgumentValueNumber(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName(), true);
    Set<List<Dimension<?>>> shapes;

    if (shapeValNum > 0) {
      OrdinalSet<InstanceKey> shapePts =
          this.getArgumentPointsToSet(
              builder, this.getShapeParameterPosition(), this.getShapeParameterName());

      if (shapePts == null || shapePts.isEmpty()) {
        LOGGER.info("No shapes found for source: " + this.getSource() + "; using default shapes.");
        shapes = this.getDefaultShapes(builder);
      } else {
        LOGGER.info(
            "Found possible shape points-to set: "
                + shapePts
                + " for source: "
                + this.getSource()
                + ".");
        shapes = this.getShapesFromShapeArgument(builder, shapePts);
      }
    } else {
      LOGGER.info("No shapes found for source: " + this.getSource() + "; using default shapes.");
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
        LOGGER.info(
            "Empty points-to set for batch_size argument in source: "
                + this.getSource()
                + "; assuming unknown.");
      } else {
        batchSizes.addAll(TensorGenerator.getPossibleLongValues(batchSizePts));
      }
    }

    if (batchSizes.isEmpty()) batchSizes.add(null);
    else
      LOGGER.info(
          "Found possible batch sizes: " + batchSizes + " for source: " + this.getSource() + ".");

    Set<List<Dimension<?>>> newShapes = HashSetFactory.make();

    for (List<Dimension<?>> shape : shapes)
      for (Long batchSize : batchSizes) {
        List<Dimension<?>> newShape = new ArrayList<>();

        // Prepend batch size.
        if (batchSize != null) newShape.add(new NumericDim(batchSize.intValue()));
        else newShape.add(null);

        newShape.addAll(shape);
        newShapes.add(newShape);
      }

    LOGGER.info("Generated shapes: " + newShapes + " for source: " + source + ".");
    return newShapes;
  }

  private void checkUnimplementedParameters(PropagationCallGraphBuilder builder) {
    Parameters[] unimplementedParameters = {
      Parameters.SPARSE, Parameters.TENSOR, Parameters.RAGGED, Parameters.TYPE_SPEC
    };

    for (Parameters p : unimplementedParameters) {
      if (this.getNumberOfPossiblePositionalArguments(builder).stream()
              .anyMatch(n -> n >= p.getIndex() + 1)
          || this.isKeywordArgumentPresent(builder, p.getName()))
        throw new UnimplementedError(
            "Unimplemented parameter " + p.getName() + " at position " + p.getIndex());
    }
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
}
