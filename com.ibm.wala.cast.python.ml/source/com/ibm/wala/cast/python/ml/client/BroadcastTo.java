package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
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

  /**
   * Parameter positions and keyword names for {@code tf.broadcast_to(input, shape, name=None)}.
   * Ordinals match the position in the XML's {@code paramNames} after the implicit {@code self}
   * receiver, so {@code Parameters.INPUT.getIndex() == 0} resolves to the first user-facing
   * positional argument.
   */
  protected enum Parameters {
    /** The tensor whose dtype is inherited by the output. */
    INPUT,

    /** The target shape (1-D integer tensor or list-of-ints) the input is broadcast to. */
    SHAPE,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in {@link #getArgumentPointsToSet} / similar arg-resolution
     * helpers when the call site uses {@code keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "input"}).
     */
    public String getName() {
      return name().toLowerCase();
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public BroadcastTo(PointsToSetVariable source) {
    super(source);
  }

  public BroadcastTo(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, Parameters.SHAPE.getIndex(), Parameters.SHAPE.getName());
    if (shapePts == null || shapePts.isEmpty()) return null;
    Set<List<Dimension<?>>> shapes;
    try {
      // `getShapesFromShapeArgument` throws `IllegalStateException` for shape forms it doesn't
      // recognize. For `BroadcastTo` specifically, a runtime-tensor shape argument is an
      // expected input pattern (e.g., `tf.broadcast_to(x, tf.shape(y))`), so we tolerate the
      // throw here and degrade to ⊤ ("tensor of unknown shape"). Other callers let the throw
      // propagate as a loud signal that modeling work is missing — see wala/ML#471's design
      // discussion on PR #245.
      shapes = this.getShapesFromShapeArgument(builder, shapePts);
    } catch (IllegalStateException e) {
      return null;
    }
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
