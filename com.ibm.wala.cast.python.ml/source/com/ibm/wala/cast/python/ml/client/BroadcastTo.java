package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.broadcast_to(input, shape, name=None)}. Output shape is the {@code shape}
 * argument's value (a 1-D int tensor / Python list of dims); output dtype follows {@code input}.
 * Same shape-from-shape-arg pattern as {@link Fill} but reads dtype from the {@code input} tensor
 * rather than a scalar value arg.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/broadcast_to">tf.broadcast_to</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class BroadcastTo extends TensorGenerator {

  protected enum Parameters {
    INPUT,
    SHAPE,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public BroadcastTo(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, Parameters.SHAPE.getIndex(), Parameters.SHAPE.getName());
    if (shapePts == null || shapePts.isEmpty()) return null;
    Set<List<Dimension<?>>> shapes = this.getShapesFromShapeArgument(builder, shapePts);
    return shapes == null || shapes.isEmpty() ? null : shapes;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int inputVn =
        this.getArgumentValueNumber(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName(), false);
    Set<DType> dtypes = this.getDTypes(builder, inputVn);
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
