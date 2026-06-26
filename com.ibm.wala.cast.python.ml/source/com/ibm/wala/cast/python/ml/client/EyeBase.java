package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Shared base for identity-matrix generators ({@code tf.eye} and {@code tf.sparse.eye}): the {@code
 * num_rows} &times; {@code num_columns} shape construction. Neither dense {@link Eye} nor {@link
 * SparseEye} is a kind of the other &mdash; dense {@code tf.eye} takes a {@code batch_shape} that
 * {@code tf.sparse.eye} lacks &mdash; so the genuine commonality lives here rather than one
 * extending the other. Replaces the inverted {@code Eye extends SparseEye} (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>). Subclasses supply their own {@code
 * dtype} parameter position, which differs by signature.
 */
public abstract class EyeBase extends TensorTypeAllocator {

  private static final int SHAPE_PARAMETER_POSITION = UNDEFINED_PARAMETER_POSITION;

  protected enum Parameters {
    NUM_ROWS,
    NUM_COLUMNS;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public EyeBase(PointsToSetVariable source) {
    super(source);
  }

  public EyeBase(CGNode node) {
    super(node);
  }

  /**
   * Retrieves the possible integer values for a specific argument.
   *
   * <p>It first checks the symbol table for constant values associated with the argument's value
   * number. If not found, it falls back to the points-to analysis results to find potential integer
   * values.
   *
   * @param builder the propagation call graph builder
   * @param paramPosition the positional index of the argument
   * @param paramName the keyword name of the argument
   * @return a set of optional integers representing the possible values. An empty optional
   *     indicates a null or non-integer value.
   */
  protected Set<Optional<Integer>> getPossibleArgumentValues(
      PropagationCallGraphBuilder builder, int paramPosition, String paramName) {
    int valNum = this.getArgumentValueNumber(builder, paramPosition, paramName, true);
    if (valNum <= 0) return HashSetFactory.make();

    if (this.getNode().getIR() != null
        && this.getNode().getIR().getSymbolTable().isConstant(valNum)) {
      Object c = this.getNode().getIR().getSymbolTable().getConstantValue(valNum);
      Set<Optional<Integer>> ret = HashSetFactory.make();
      if (c instanceof Number) {
        ret.add(Optional.of(((Number) c).intValue()));
      } else if (c == null) {
        ret.add(Optional.empty());
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }

    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPosition, paramName);

    if (pts == null || pts.isEmpty())
      // Fallback to default (empty).
      return HashSetFactory.make();

    return StreamSupport.stream(pts.spliterator(), false)
        .map(TensorTypeAllocator::getIntValueFromInstanceKey)
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

            shape.add(axisDim(nRow));
            shape.add(axisDim(nCol2));

            ret.add(shape);
          }
        } else {
          List<Dimension<?>> shape = new ArrayList<>();

          shape.add(axisDim(nRow));
          shape.add(axisDim(nCol));

          ret.add(shape);
        }
    }

    return ret;
  }

  private Set<Optional<Integer>> getNumberOfRows(PropagationCallGraphBuilder builder) {
    Set<Optional<Integer>> values =
        this.getPossibleArgumentValues(
            builder, this.getNumRowsParameterPosition(), this.getNumRowsParameterName());

    // num_rows is mandatory, but when it's unresolvable (content-dependent) don't abort the whole
    // analysis: treat it as a single unknown value so the shape floors to a dynamic rank-2 tensor
    // rather than throwing. wala/ML#611.
    if (values == null || values.isEmpty()) return Set.of(Optional.empty());

    return values;
  }

  /**
   * Maps a possibly-unknown axis size to a dimension: a {@link NumericDim} when the value is known,
   * a {@link DynamicDim} when it isn't (so an unresolvable {@code num_rows}/{@code num_columns}
   * floors to a dynamic axis rather than crashing on {@link Optional#get()}). wala/ML#611.
   *
   * @param value The possibly-unknown axis size.
   * @return A {@link NumericDim} if present, else {@link DynamicDim#INSTANCE}.
   */
  private static Dimension<?> axisDim(Optional<Integer> value) {
    return value.isPresent() ? new NumericDim(value.get()) : DynamicDim.INSTANCE;
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

  protected int getNumColumnsArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getNumColumnsParameterPosition(), this.getNumColumnsParameterName(), true);
  }
}
