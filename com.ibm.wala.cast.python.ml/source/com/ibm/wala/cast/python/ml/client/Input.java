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
import java.util.EnumSet;
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

  private static final int BATCH_SIZE_PARAMETER_POSITION = 1;

  private static final int DTYPE_PARAMETER_POSITION = 3;

  private static final int SPARSE_PARAMETER_POSITION = 4;

  private static final int TENSOR_PARAMETER_POSITION = 5;

  private static final int RAGGED_PARAMETER_POSITION = 6;

  private static final int TYPE_SPEC_PARAMETER_POSITION = 7;

  public Input(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int valNum = getArgumentValueNumber(builder, this.getDTypeParameterPosition(), "dtype", true);

    if (valNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), valNum);
      OrdinalSet<InstanceKey> pointsToSet = pa.getPointsToSet(pk);

      if (pointsToSet != null && !pointsToSet.isEmpty()) {
        LOGGER.info("Found possible dtypes: " + pointsToSet + " for source: " + source + ".");
        return getDTypesFromDTypeArgument(builder, pointsToSet);
      }
    }

    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    checkUnimplementedParameters(builder);

    int shapeValNum =
        getArgumentValueNumber(builder, this.getShapeParameterPosition(), "shape", true);

    Set<List<Dimension<?>>> shapes;

    if (shapeValNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), shapeValNum);
      OrdinalSet<InstanceKey> shapePts = pa.getPointsToSet(pk);

      if (shapePts != null && !shapePts.isEmpty()) {
        LOGGER.info(
            "Found possible shape points-to set: " + shapePts + " for source: " + source + ".");
        shapes = getShapesFromShapeArgument(builder, shapePts);
      } else {
        LOGGER.info("No shapes found for source: " + source + "; using default shapes.");
        shapes = getDefaultShapes(builder);
      }
    } else {
      LOGGER.info("No shapes found for source: " + source + "; using default shapes.");
      shapes = getDefaultShapes(builder);
    }

    // Handle `batch_size`.
    int batchSizeValNum =
        getArgumentValueNumber(builder, BATCH_SIZE_PARAMETER_POSITION, "batch_size", true);

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
    int[] unimplementedPositionalArgs = {
      SPARSE_PARAMETER_POSITION,
      TENSOR_PARAMETER_POSITION,
      RAGGED_PARAMETER_POSITION,
      TYPE_SPEC_PARAMETER_POSITION
    };

    String[] unimplementedKeywords = {"sparse", "tensor", "ragged", "type_spec"};

    for (int i = 0; i < unimplementedPositionalArgs.length; i++) {
      int pos = unimplementedPositionalArgs[i];
      String kw = unimplementedKeywords[i];

      int valNum = getArgumentValueNumber(builder, pos, kw, true);
      if (valNum > 0) throw new UnimplementedError("Unimplemented argument: " + kw);
    }
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
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
