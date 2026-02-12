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
 * A representation of the `tf.RaggedTensor.from_row_lengths` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_lengths">tf.RaggedTensor.from_row_lengths</a>.
 */
public class RaggedFromRowLengths extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromRowLengths.class.getName());

  protected enum Parameters {
    VALUES,
    ROW_LENGTHS,
    NAME,
    VALIDATE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public RaggedFromRowLengths(PointsToSetVariable source) {
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

  protected int getRowLengthsParameterPosition() {
    return Parameters.ROW_LENGTHS.getIndex();
  }

  protected String getRowLengthsParameterName() {
    return Parameters.ROW_LENGTHS.getName();
  }

  protected OrdinalSet<InstanceKey> getRowLengthsPointsToSet(PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getRowLengthsParameterPosition(), getRowLengthsParameterName());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Determine `nrows` from `row_lengths`.
    // The number of rows is len(row_lengths).
    OrdinalSet<InstanceKey> rowLengthsPts = this.getRowLengthsPointsToSet(builder);
    Set<List<Dimension<?>>> rowLengthsShapes = emptySet();
    if (rowLengthsPts != null && !rowLengthsPts.isEmpty()) {
      rowLengthsShapes = this.getShapesOfValue(builder, rowLengthsPts);
    }

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    if (!rowLengthsShapes.isEmpty()) {
      for (List<Dimension<?>> shape : rowLengthsShapes) {
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
