package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class Eye extends EyeBase {

  protected enum Parameters {
    NUM_ROWS,
    NUM_COLUMNS,
    BATCH_SHAPE,
    DTYPE,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Eye(PointsToSetVariable source) {
    super(source);
  }

  public Eye(CGNode node) {
    super(node);
  }

  protected int getBatchShapesArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getBatchShapeParameterPosition(), this.getBatchShapeParameterName(), true);
  }

  protected int getBatchShapeParameterPosition() {
    return Parameters.BATCH_SHAPE.getIndex();
  }

  protected String getBatchShapeParameterName() {
    return Parameters.BATCH_SHAPE.getName();
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = super.getShapes(builder);
    Set<List<Dimension<?>>> batchShapes = this.getBatchShapes(builder);

    if (batchShapes == null)
      // `batch_shape` is present but unresolvable (content-dependent), so the number of leading
      // batch dimensions is unknown and the overall rank can't be known: floor to ⊤ rather than
      // throwing (which aborted the whole analysis). wala/ML#611.
      return null;

    if (batchShapes.isEmpty()) return ret;

    // Prepend each batch shape to each base shape, building a fresh list per combination. Mutating
    // the lists returned by super.getShapes would alias shared shape lists, and re-prepending per
    // batch shape would stack multiple batch shapes onto the same list (double-prepend).
    // wala/ML#591.
    Set<List<Dimension<?>>> withBatchShapes = HashSetFactory.make();
    for (List<Dimension<?>> batchDim : batchShapes)
      for (List<Dimension<?>> retDim : ret) {
        List<Dimension<?>> combined = new ArrayList<>(batchDim);
        combined.addAll(retDim);
        withBatchShapes.add(combined);
      }

    return withBatchShapes;
  }

  /**
   * Returns the possible leading {@code batch_shape} dimensions to prepend to the identity matrix.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return An empty set when {@code batch_shape} is absent (no leading dimensions); the resolved
   *     batch shapes when it is present and resolvable; or {@code null} (⊤) when it is present but
   *     unresolvable (content-dependent), since the leading rank is then unknown. wala/ML#611.
   */
  private Set<List<Dimension<?>>> getBatchShapes(PropagationCallGraphBuilder builder) {
    int valNum = this.getBatchShapesArgumentValueNumber(builder);
    if (valNum <= 0) return emptySet();

    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, this.getBatchShapeParameterPosition(), this.getBatchShapeParameterName());

    if (pts == null || pts.isEmpty()) return emptySet();

    Set<List<Dimension<?>>> shapesFromShapeArgument = this.getShapesFromShapeArgument(builder, pts);

    if (shapesFromShapeArgument == null || shapesFromShapeArgument.isEmpty())
      // `batch_shape` is present but its contents are unresolvable (content-dependent). Signal ⊤ to
      // the caller with `null` rather than throwing; an empty set would instead mean "no batch
      // shape" and wrongly drop the (definitely-present) leading dimensions. wala/ML#611.
      return null;

    return shapesFromShapeArgument;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
