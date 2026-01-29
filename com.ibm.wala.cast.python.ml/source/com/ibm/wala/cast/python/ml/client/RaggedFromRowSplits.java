package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromRowSplits.Parameters.ROW_SPLITS;
import static com.ibm.wala.cast.python.ml.client.RaggedFromRowSplits.Parameters.VALUES;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.RaggedTensor.from_row_splits` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_splits">tf.RaggedTensor.from_row_splits</a>.
 */
public class RaggedFromRowSplits extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromRowSplits.class.getName());

  protected enum Parameters {
    VALUES,
    ROW_SPLITS,
    NAME,
    VALIDATE;

    public String getParameterName() {
      return name().toLowerCase();
    }
  }

  public RaggedFromRowSplits(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return VALUES.ordinal();
  }

  @Override
  protected String getValuesParameterName() {
    return VALUES.getParameterName();
  }

  protected int getRowSplitsParameterPosition() {
    return ROW_SPLITS.ordinal();
  }

  protected String getRowSplitsParameterName() {
    return ROW_SPLITS.getParameterName();
  }

  protected OrdinalSet<InstanceKey> getRowSplitsPointsToSet(PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getRowSplitsParameterPosition(), getRowSplitsParameterName());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Determine `nrows` from `row_splits`.
    // The number of rows is len(row_splits) - 1.
    OrdinalSet<InstanceKey> rowSplitsPts = this.getRowSplitsPointsToSet(builder);
    Set<List<Dimension<?>>> rowSplitsShapes = emptySet();
    if (rowSplitsPts != null && !rowSplitsPts.isEmpty()) {
      rowSplitsShapes = this.getShapesOfValue(builder, rowSplitsPts);
    }

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    if (!rowSplitsShapes.isEmpty()) {
      for (List<Dimension<?>> shape : rowSplitsShapes) {
        if (!shape.isEmpty()) {
          Dimension<?> firstDim = shape.get(0);
          if (firstDim instanceof NumericDim) {
            int len = ((NumericDim) firstDim).value();
            possibleRowDims.add(new NumericDim(Math.max(0, len - 1)));
          } else {
            possibleRowDims.add(null);
          }
        }
      }
    } else {
      possibleRowDims.add(null);
    }

    final Set<Dimension<?>> finalPossibleRowDims = possibleRowDims;
    LOGGER.info(() -> "Inferred nrows for " + this.getSource() + ": " + finalPossibleRowDims + ".");

    // 2. Determine shape of `values`.
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, getValuesParameterPosition(), getValuesParameterName());
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesPts != null && !valuesPts.isEmpty()) {
      valuesShapes = this.getShapesOfValue(builder, valuesPts);
    }

    final Set<List<Dimension<?>>> finalValuesShapes = valuesShapes;
    LOGGER.info(
        () -> "Possible values shapes for " + this.getSource() + ": " + finalValuesShapes + ".");

    return constructRaggedShape(possibleRowDims, valuesShapes);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION; // No explicit shape argument
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION; // No explicit dtype argument
  }
}
