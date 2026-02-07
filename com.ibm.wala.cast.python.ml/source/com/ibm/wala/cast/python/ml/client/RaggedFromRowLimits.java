package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.RaggedTensor.from_row_limits` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_limits">tf.RaggedTensor.from_row_limits</a>.
 */
public class RaggedFromRowLimits extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromRowLimits.class.getName());

  protected enum Parameters {
    VALUES,
    ROW_LIMITS,
    NAME,
    VALIDATE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public RaggedFromRowLimits(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return Parameters.VALUES.getIndex();
  }

  @Override
  protected String getValuesParameterName() {
    return Parameters.VALUES.getName();
  }

  protected int getRowLimitsParameterPosition() {
    return Parameters.ROW_LIMITS.getIndex();
  }

  protected String getRowLimitsParameterName() {
    return Parameters.ROW_LIMITS.getName();
  }

  protected OrdinalSet<InstanceKey> getRowLimitsPointsToSet(PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getRowLimitsParameterPosition(), getRowLimitsParameterName());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Determine `nrows` from `row_limits`.
    // The number of rows is len(row_limits).
    OrdinalSet<InstanceKey> rowLimitsPts = this.getRowLimitsPointsToSet(builder);
    Set<List<Dimension<?>>> rowLimitsShapes = emptySet();
    if (rowLimitsPts != null && !rowLimitsPts.isEmpty()) {
      rowLimitsShapes = this.getShapesOfValue(builder, rowLimitsPts);
    }

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    if (!rowLimitsShapes.isEmpty()) {
      for (List<Dimension<?>> shape : rowLimitsShapes) {
        if (!shape.isEmpty()) {
          possibleRowDims.add(shape.get(0));
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

    return constructRaggedShape(possibleRowDims, valuesShapes);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
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
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
