package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromRowStarts.Parameters.ROW_STARTS;
import static com.ibm.wala.cast.python.ml.client.RaggedFromRowStarts.Parameters.VALUES;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.RaggedTensor.from_row_starts` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_starts">tf.RaggedTensor.from_row_starts</a>.
 */
public class RaggedFromRowStarts extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromRowStarts.class.getName());

  private static final String VALUES_PARAM = "values";

  private static final String ROW_STARTS_PARAM = "row_starts";

  protected enum Parameters {
    VALUES,
    ROW_STARTS,
    NAME,
    VALIDATE
  }

  public RaggedFromRowStarts(PointsToSetVariable source) {
    super(source);
  }

  protected int getValuesParameterPosition() {
    return VALUES.ordinal();
  }

  protected int getValuesArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getValuesParameterPosition(), VALUES_PARAM, true);
  }

  protected int getRowStartsParameterPosition() {
    return ROW_STARTS.ordinal();
  }

  protected int getRowStartsArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getRowStartsParameterPosition(), ROW_STARTS_PARAM, true);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // 1. Determine `nrows` from `row_starts`.
    // The number of rows is len(row_starts) - 1.
    int rowStartsValNum = this.getRowStartsArgumentValueNumber(builder);
    Set<List<Dimension<?>>> rowStartsShapes = emptySet();
    if (rowStartsValNum > 0) {
      rowStartsShapes = this.getShapes(builder, rowStartsValNum);
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
        } else {
          possibleRowDims.add(null);
        }
      }
    } else {
      possibleRowDims.add(null);
    }

    final Set<Dimension<?>> finalPossibleRowDims = possibleRowDims;
    LOGGER.info(() -> "Inferred nrows for " + this.getSource() + ": " + finalPossibleRowDims + ".");

    // 2. Determine shape of `values`.
    int valuesValNum = this.getValuesArgumentValueNumber(builder);
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesValNum > 0) {
      valuesShapes = this.getShapes(builder, valuesValNum);
    }

    final Set<List<Dimension<?>>> finalValuesShapes = valuesShapes;
    LOGGER.info(
        () -> "Possible values shapes for " + this.getSource() + ": " + finalValuesShapes + ".");

    if (valuesShapes.isEmpty()) {
      for (Dimension<?> rowDim : possibleRowDims) {
        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(rowDim);
        shape.add(null); // Ragged dimension
        ret.add(shape);
      }
      LOGGER.info(
          () -> "Determined default ragged shapes for " + this.getSource() + ": " + ret + ".");
      return ret;
    }

    // 3. Construct result shape: [nrows, (ragged)] + values.shape[1:]
    for (Dimension<?> rowDim : possibleRowDims) {
      for (List<Dimension<?>> valShape : valuesShapes) {
        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(rowDim);
        shape.add(null); // Ragged dimension

        if (valShape.size() > 1) {
          shape.addAll(valShape.subList(1, valShape.size()));
        }
        ret.add(shape);
      }
    }

    LOGGER.info(() -> "Determined final ragged shapes for " + this.getSource() + ": " + ret + ".");
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected int getShapeParameterPosition() {
    return -1; // No explicit shape argument
  }

  @Override
  protected int getDTypeParameterPosition() {
    return -1; // No explicit dtype argument
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Infer from values
    int valuesValNum = this.getValuesArgumentValueNumber(builder);
    if (valuesValNum > 0) {
      EnumSet<DType> ret = this.getDTypes(builder, valuesValNum);
      LOGGER.info(() -> "Inferred dtypes from values for " + this.getSource() + ": " + ret + ".");
      return ret;
    }
    return EnumSet.noneOf(DType.class);
  }
}
