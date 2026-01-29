package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class SparseEye extends Ones {

  private static final int SHAPE_PARAMETER_POSITION = UNDEFINED_PARAMETER_POSITION;

  protected enum Parameters {
    NUM_ROWS,
    NUM_COLUMNS,
    DTYPE,
    NAME;

    public String getParameterName() {
      return name().toLowerCase();
    }

    public int getParameterIndex() {
      return ordinal();
    }
  }

  public SparseEye(PointsToSetVariable source) {
    super(source);
  }

  private Set<Optional<Integer>> getPossiblePositionalArgumentValues(
      PropagationCallGraphBuilder builder, int paramPosition, String paramName) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    Set<Integer> possibleNumArgs = this.getNumberOfPossiblePositionalArguments(builder);

    return possibleNumArgs.stream()
        .filter(numArgs -> numArgs >= paramPosition + 1)
        .map(
            _ -> {
              int argValNum = this.getArgumentValueNumber(builder, paramPosition, paramName, true);
              if (argValNum <= 0) return null;

              PointerKey pointerKey =
                  pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), argValNum);
              return pointerAnalysis.getPointsToSet(pointerKey);
            })
        .filter(pts -> pts != null)
        .flatMap(pts -> StreamSupport.stream(pts.spliterator(), false))
        .map(SparseEye::getIntValueFromInstanceKey)
        .collect(Collectors.toSet());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Set<Optional<Integer>> numRows = this.getNumberOfRows(builder);
    Set<Optional<Integer>> numColumns = this.getNumberOfColumns(builder);

    for (Optional<Integer> nRow : numRows) {
      if (numColumns.isEmpty())
        // If numColumns is not provided, it defaults to numRows.
        for (Optional<Integer> nCol : numRows)
          // Build the shape using nRow and nCol.
          numColumns.add(nCol);

      for (Optional<Integer> nCol : numColumns)
        if (nCol.isEmpty()) {
          // If numColumns is not provided, it defaults to numRows.
          for (Optional<Integer> nCol2 : numRows) {
            // Build the shape using nRow and nCol.
            List<Dimension<?>> shape = new ArrayList<>();

            NumericDim rowDim = new NumericDim(nRow.get());
            NumericDim colDim = new NumericDim(nCol2.get());

            shape.add(rowDim);
            shape.add(colDim);

            ret.add(shape);
          }
        } else {
          List<Dimension<?>> shape = new ArrayList<>();

          NumericDim rowDim = new NumericDim(nRow.get());
          NumericDim colDim = new NumericDim(nCol.get());

          shape.add(rowDim);
          shape.add(colDim);

          ret.add(shape);
        }
    }

    return ret;
  }

  private Set<Optional<Integer>> getNumberOfRows(PropagationCallGraphBuilder builder) {
    Set<Optional<Integer>> values =
        this.getPossiblePositionalArgumentValues(
            builder, this.getNumRowsParameterPosition(), this.getNumRowsParameterName());

    if (values == null || values.isEmpty())
      throw new IllegalStateException("The num_rows parameter is required for tf.eye().");

    return values;
  }

  private Set<Optional<Integer>> getNumberOfColumns(PropagationCallGraphBuilder builder) {
    return this.getPossiblePositionalArgumentValues(
        builder, this.getNumColumnsParameterPosition(), this.getNumColumnsParameterName());
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  protected int getNumRowsParameterPosition() {
    return Parameters.NUM_ROWS.getParameterIndex();
  }

  protected String getNumRowsParameterName() {
    return Parameters.NUM_ROWS.getParameterName();
  }

  protected int getNumRowsArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getNumRowsParameterPosition());
  }

  protected int getNumColumnsParameterPosition() {
    return Parameters.NUM_COLUMNS.getParameterIndex();
  }

  protected String getNumColumnsParameterName() {
    return Parameters.NUM_COLUMNS.getParameterName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getParameterIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getParameterName();
  }

  protected int getNumColumnsArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getNumColumnsParameterPosition(), this.getNumColumnsParameterName(), true);
  }
}
