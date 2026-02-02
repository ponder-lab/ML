package com.ibm.wala.cast.python.ml.client;

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
 * A representation of the `tf.RaggedTensor.from_row_starts` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_starts">tf.RaggedTensor.from_row_starts</a>.
 */
public class RaggedFromRowStarts extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromRowStarts.class.getName());

  protected enum Parameters {
    VALUES,
    ROW_STARTS,
    NAME,
    VALIDATE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public RaggedFromRowStarts(PointsToSetVariable source) {
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

  protected int getRowStartsParameterPosition() {
    return Parameters.ROW_STARTS.getIndex();
  }

  protected String getRowStartsParameterName() {
    return Parameters.ROW_STARTS.getName();
  }

  protected OrdinalSet<InstanceKey> getRowStartsPointsToSet(PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getRowStartsParameterPosition(), getRowStartsParameterName());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Determine `nrows` from `row_starts`.
    // The number of rows is len(row_starts).
    OrdinalSet<InstanceKey> rowStartsPts = this.getRowStartsPointsToSet(builder);
    Set<List<Dimension<?>>> rowStartsShapes = emptySet();
    if (rowStartsPts != null && !rowStartsPts.isEmpty()) {
      rowStartsShapes = this.getShapesOfValue(builder, rowStartsPts);
    }

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    if (!rowStartsShapes.isEmpty()) {
      for (List<Dimension<?>> shape : rowStartsShapes) {
        if (!shape.isEmpty()) {
          Dimension<?> firstDim = shape.get(0);
          if (firstDim instanceof NumericDim) {
            int len = ((NumericDim) firstDim).value();
            possibleRowDims.add(new NumericDim(len));
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
    OrdinalSet<InstanceKey> valuesPts = this.getValuesPointsToSet(builder);
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
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION; // No explicit dtype argument
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
