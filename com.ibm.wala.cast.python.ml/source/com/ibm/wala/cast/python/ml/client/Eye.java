package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.List;
import java.util.Set;

public class Eye extends SparseEye {

  protected enum Parameters {
    NUM_ROWS,
    NUM_COLUMNS,
    BATCH_SHAPE,
    DTYPE,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Eye(PointsToSetVariable source) {
    super(source);
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

    // prepend batch dimensions to each shape.
    for (List<Dimension<?>> batchDim : batchShapes)
      for (List<Dimension<?>> retDim : ret) retDim.addAll(0, batchDim);

    return ret;
  }

  private Set<List<Dimension<?>>> getBatchShapes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder,
            this.getBatchShapeParameterPosition(),
            this.getBatchShapeParameterName(),
            true);
    if (valNum <= 0) return emptySet();

    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, this.getBatchShapeParameterPosition(), this.getBatchShapeParameterName());

    if (pts == null || pts.isEmpty())
      // Fallback to default (empty).
      return emptySet();

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
