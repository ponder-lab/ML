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

  private Set<List<Dimension<?>>> getBatchShapes(PropagationCallGraphBuilder builder) {
    int valNum = this.getBatchShapesArgumentValueNumber(builder);
    if (valNum <= 0) return emptySet();

    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, this.getBatchShapeParameterPosition(), this.getBatchShapeParameterName());

    if (pts == null || pts.isEmpty()) return emptySet();

    Set<List<Dimension<?>>> shapesFromShapeArgument = this.getShapesFromShapeArgument(builder, pts);

    if (shapesFromShapeArgument == null || shapesFromShapeArgument.isEmpty())
      throw new IllegalStateException(
          "Batch shape argument for tf.eye() should be a list of dimensions.");

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
