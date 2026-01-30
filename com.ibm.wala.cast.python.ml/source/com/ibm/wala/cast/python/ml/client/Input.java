package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.HashSet;
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

    public String getParameterName() {
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

    OrdinalSet<InstanceKey> pointsToSet = null;

    if (valNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), valNum);
      pointsToSet = pa.getPointsToSet(pk);
    }

    if (pointsToSet != null && !pointsToSet.isEmpty()) {
      LOGGER.info("Found possible dtypes: " + pointsToSet + " for source: " + source + ".");
      return getDTypesFromDTypeArgument(builder, pointsToSet);
    }

    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    checkUnimplementedParameters(builder);

    int shapeValNum =
        this.getArgumentValueNumber(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName(), true);

    OrdinalSet<InstanceKey> shapePts = null;

    if (shapeValNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), shapeValNum);
      shapePts = pa.getPointsToSet(pk);
    }

    Set<List<Dimension<?>>> shapes;

    if (shapePts != null && !shapePts.isEmpty()) {
      LOGGER.info(
          "Found possible shape points-to set: " + shapePts + " for source: " + source + ".");
      shapes = getShapesFromShapeArgument(builder, shapePts);
    } else {
      LOGGER.info("No shapes found for source: " + source + "; using default shapes.");
      shapes = getDefaultShapes(builder);
    }

    // Handle `batch_size`.
    int batchSizeValNum =
        this.getArgumentValueNumber(
            builder, this.getBatchSizeParameterPosition(), this.getBatchSizeParameterName(), true);

    Set<Long> batchSizes = new HashSet<>();

    if (batchSizeValNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), batchSizeValNum);
      OrdinalSet<InstanceKey> pts = pa.getPointsToSet(pk);
      batchSizes.addAll(getPossibleLongArguments(pts));
    }

    if (batchSizes.isEmpty()) batchSizes.add(null);
    else LOGGER.info("Found possible batch sizes: " + batchSizes + " for source: " + source + ".");

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
      int valNum = this.getArgumentValueNumber(builder, p.getIndex(), p.getParameterName(), true);
      if (valNum > 0)
        throw new UnimplementedError(
            "Unimplemented parameter " + p.getParameterName() + " at position " + p.getIndex());
    }
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  protected String getShapeParameterName() {
    return Parameters.SHAPE.getParameterName();
  }

  protected int getBatchSizeParameterPosition() {
    return Parameters.BATCH_SIZE.getIndex();
  }

  protected String getBatchSizeParameterName() {
    return Parameters.BATCH_SIZE.getParameterName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getParameterName();
  }

  private static Set<Long> getPossibleLongArguments(OrdinalSet<InstanceKey> pointsToSet) {
    Set<Long> ret = HashSetFactory.make();

    if (pointsToSet == null) return ret;

    for (InstanceKey instanceKey : pointsToSet)
      if (instanceKey instanceof ConstantKey) {
        ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
        Object constantKeyValue = constantKey.getValue();

        if (constantKeyValue instanceof Long) ret.add((Long) constantKeyValue);
        else if (constantKeyValue instanceof Integer)
          ret.add(((Integer) constantKeyValue).longValue());
        else if (constantKeyValue == null) ret.add(null);
        else
          throw new IllegalStateException(
              "Expected Long or Integer constant for batch size, but got: "
                  + constantKeyValue.getClass());
      } else
        throw new IllegalStateException(
            "Expected ConstantKey for batch size, but got: " + instanceKey.getClass());

    return ret;
  }
}
