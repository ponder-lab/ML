package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Eye.Parameters.BATCH_SHAPE;
import static com.ibm.wala.cast.python.ml.client.Eye.Parameters.DTYPE;
import static com.ibm.wala.cast.python.ml.client.Eye.Parameters.NUM_COLUMNS;
import static com.ibm.wala.cast.python.ml.client.Eye.Parameters.NUM_ROWS;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
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

public class Eye extends Ones {

  private static final String FUNCTION_NAME = "tf.eye()";

  private static final int SHAPE_PARAMETER_POSITION = -1;

  enum Parameters {
    NUM_ROWS,
    NUM_COLUMNS,
    BATCH_SHAPE,
    DTYPE,
    NAME
  }

  public Eye(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  protected int getNumRowsParameterPosition() {
    return NUM_ROWS.ordinal();
  }

  protected int getNumRowsArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getNumRowsParameterPosition());
  }

  protected int getBatchShapesArgumentValueNumber() {
    // TOOD: Handle keyword arguments.
    return this.getArgumentValueNumber(this.getBatchShapeParameterPosition());
  }

  protected int getNumColumnsParameterPosition() {
    return NUM_COLUMNS.ordinal();
  }

  protected int getNumColumnsArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(this.getNumColumnsParameterPosition());
  }

  protected int getBatchShapeParameterPosition() {
    return BATCH_SHAPE.ordinal();
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

    Set<List<Dimension<?>>> batchShapes = this.getBatchShapes(builder);

    // prepend batch dimensions to each shape.
    for (List<Dimension<?>> batchDim : batchShapes)
      for (List<Dimension<?>> retDim : ret) retDim.addAll(0, batchDim);

    return ret;
  }

  private Set<Optional<Integer>> getNumberOfRows(PropagationCallGraphBuilder builder) {
    // TODO Handle keyword arguments.
    Set<Optional<Integer>> values =
        this.getPossiblePositionalArgumentValues(builder, this.getNumRowsParameterPosition());

    if (values == null || values.isEmpty())
      throw new IllegalStateException("The num_rows parameter is required for tf.eye().");

    return values;
  }

  private Set<Optional<Integer>> getNumberOfColumns(PropagationCallGraphBuilder builder) {
    // TODO Handle keyword arguments.
    return this.getPossiblePositionalArgumentValues(builder, this.getNumColumnsParameterPosition());
  }

  private Set<List<Dimension<?>>> getBatchShapes(PropagationCallGraphBuilder builder) {
    // TODO Handle keyword arguments.
    Set<Integer> possibleNumArgs = this.getNumberOfPossiblePositionalArguments(builder);

    if (possibleNumArgs.contains(this.getBatchShapeParameterPosition() + 1)) {
      PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

      PointerKey pointerKey =
          pointerAnalysis
              .getHeapModel()
              .getPointerKeyForLocal(this.getNode(), this.getBatchShapesArgumentValueNumber());

      OrdinalSet<InstanceKey> pts = pointerAnalysis.getPointsToSet(pointerKey);

      Set<List<Dimension<?>>> shapesFromShapeArgument =
          this.getShapesFromShapeArgument(builder, pts);

      if (shapesFromShapeArgument == null || shapesFromShapeArgument.isEmpty())
        throw new IllegalStateException(
            "Batch shape argument for tf.eye() should be a list of dimensions.");

      return shapesFromShapeArgument;
    }

    return emptySet();
  }

  private Set<Optional<Integer>> getPossiblePositionalArgumentValues(
      PropagationCallGraphBuilder builder, int paramPosition) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    Set<Integer> possibleNumArgs = this.getNumberOfPossiblePositionalArguments(builder);

    return possibleNumArgs.stream()
        .filter(numArgs -> numArgs >= paramPosition + 1)
        .map(
            _ -> {
              PointerKey pointerKey =
                  pointerAnalysis
                      .getHeapModel()
                      .getPointerKeyForLocal(
                          this.getNode(), this.getArgumentValueNumber(paramPosition));
              return pointerAnalysis.getPointsToSet(pointerKey);
            })
        .flatMap(pts -> StreamSupport.stream(pts.spliterator(), false))
        .map(Eye::getIntValueFromInstanceKey)
        .collect(Collectors.toSet());
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }
}
