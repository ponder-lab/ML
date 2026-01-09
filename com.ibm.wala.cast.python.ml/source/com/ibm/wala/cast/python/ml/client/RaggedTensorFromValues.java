package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A base class for ragged tensor generators that are constructed from a values tensor.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class RaggedTensorFromValues extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(RaggedTensorFromValues.class.getName());

  protected static final String VALUES_PARAM = "values";

  public RaggedTensorFromValues(PointsToSetVariable source) {
    super(source);
  }

  protected abstract int getValuesParameterPosition();

  protected int getValuesArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getValuesParameterPosition(), VALUES_PARAM, true);
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

  protected Set<List<Dimension<?>>> constructRaggedShape(
      Set<Dimension<?>> possibleRowDims, Set<List<Dimension<?>>> valuesShapes) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

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

    // Construct result shape: [nrows, (ragged)] + values.shape[1:]
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
}
