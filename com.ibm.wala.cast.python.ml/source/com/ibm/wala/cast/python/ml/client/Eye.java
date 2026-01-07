package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Eye.Parameters.BATCH_SHAPE;
import static com.ibm.wala.cast.python.ml.client.Eye.Parameters.DTYPE;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
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
    NAME
  }

  public Eye(PointsToSetVariable source) {
    super(source);
  }

  protected int getBatchShapesArgumentValueNumber() {
    // TOOD: Handle keyword arguments.
    return this.getArgumentValueNumber(this.getBatchShapeParameterPosition());
  }

  protected int getBatchShapeParameterPosition() {
    return BATCH_SHAPE.ordinal();
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

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }
}
