package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;

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

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // A starred unpack — `np.eye(*shape)` — lands the whole tuple in the num_rows slot instead of
    // separate `N` and `M` scalars, so the per-argument integer read fails on it and the shape
    // floors to ⊤. `np.eye(N, M)` has shape `(N, M)`, so the tuple's first two elements ARE the
    // shape (`M` defaulting to `N`); read them from the tuple. wala/ML#910.
    Set<List<Dimension<?>>> starred = this.getStarredShapes(builder);
    if (starred != null) return starred;

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

  /**
   * Resolves the shape when the identity-matrix call is a starred unpack of a sequence, e.g. {@code
   * np.eye(*shape)} where {@code shape} is a tuple local. The unpack places the whole sequence in
   * the {@code num_rows} slot rather than separate {@code N} and {@code M} scalars, so the scalar
   * integer read on that slot cannot fold it and the shape floors to ⊤ (wala/ML#910). Since {@code
   * np.eye(N, M)} produces a shape of exactly {@code (N, M)}, the sequence's first two elements are
   * the shape, with {@code M} defaulting to {@code N} for a one-element sequence.
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return The resolved shapes, or {@code null} when the {@code num_rows} argument is not a
   *     sequence (the ordinary scalar path applies) or the sequence's elements do not resolve.
   */
  private Set<List<Dimension<?>>> getStarredShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, this.getNumRowsParameterPosition(), this.getNumRowsParameterName());
    if (pts == null || pts.isEmpty()) return null;

    // Only a sequence in the num_rows slot is a starred unpack; a scalar takes the ordinary path.
    boolean isSequence = false;
    for (InstanceKey instanceKey : pts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      if (asin == null) continue;
      TypeReference reference = asin.concreteType().getReference();
      if (reference.equals(list) || reference.equals(tuple)) {
        isSequence = true;
        break;
      }
    }
    if (!isSequence) return null;

    Set<List<Dimension<?>>> sequenceShapes = this.getShapesFromShapeArgument(builder, pts);
    if (sequenceShapes == null || sequenceShapes.isEmpty()) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> dims : sequenceShapes) {
      if (dims.isEmpty()) continue;
      List<Dimension<?>> shape = new ArrayList<>();
      shape.add(dims.get(0));
      shape.add(dims.size() >= 2 ? dims.get(1) : dims.get(0));
      ret.add(shape);
    }
    return ret.isEmpty() ? null : ret;
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
   * an {@link UnresolvedDim} when it isn't (so an unresolvable {@code num_rows}/{@code num_columns}
   * floors to an unknown fixed axis rather than crashing on {@link Optional#get()}, wala/ML#611) —
   * the argument is a Python scalar, so the runtime size is a fixed value the analysis could not
   * compute (wala/ML#721).
   *
   * @param value The possibly-unknown axis size.
   * @return A {@link NumericDim} if present, else {@link UnresolvedDim#INSTANCE}.
   */
  private static Dimension<?> axisDim(Optional<Integer> value) {
    return value.isPresent() ? new NumericDim(value.get()) : UnresolvedDim.INSTANCE;
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
