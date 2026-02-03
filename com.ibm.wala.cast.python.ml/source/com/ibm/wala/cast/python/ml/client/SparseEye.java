package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
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

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseEye(PointsToSetVariable source) {
    super(source);
  }

  private Set<Optional<Integer>> getPossibleArgumentValues(
      PropagationCallGraphBuilder builder, int paramPosition, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPosition, paramName);

    if (pts == null || pts.isEmpty()) return HashSetFactory.make();

    return StreamSupport.stream(pts.spliterator(), false)
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
        this.getPossibleArgumentValues(
            builder, this.getNumRowsParameterPosition(), this.getNumRowsParameterName());

    if (values == null || values.isEmpty())
      throw new IllegalStateException("The num_rows parameter is required for tf.eye().");

    return values;
  }

  private Set<Optional<Integer>> getNumberOfColumns(PropagationCallGraphBuilder builder) {
    return this.getPossibleArgumentValues(
        builder, this.getNumColumnsParameterPosition(), this.getNumColumnsParameterName());
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  protected int getNumRowsParameterPosition() {
    return Parameters.NUM_ROWS.getIndex();
  }

  protected String getNumRowsParameterName() {
    return Parameters.NUM_ROWS.getName();
  }

  protected int getNumRowsArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getNumRowsParameterPosition());
  }

  protected int getNumColumnsParameterPosition() {
    return Parameters.NUM_COLUMNS.getIndex();
  }

  protected String getNumColumnsParameterName() {
    return Parameters.NUM_COLUMNS.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  protected int getNumColumnsArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getNumColumnsParameterPosition(), this.getNumColumnsParameterName(), true);
  }
}
