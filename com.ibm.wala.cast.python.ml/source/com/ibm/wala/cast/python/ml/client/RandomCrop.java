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
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.image.random_crop(value, size, seed=None, name=None)}. The result has the
 * shape the {@code size} argument gives and the dtype of {@code value}, the same form as {@link
 * BroadcastTo}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/image/random_crop">
 *     tf.image.random_crop</a>
 */
public class RandomCrop extends TensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.image.random_crop}, after the implicit
   * {@code self} receiver.
   */
  protected enum Parameters {
    /** The tensor cropped, whose dtype the result has. */
    VALUE,

    /** The shape of the crop, a 1-D integer tensor or a list of integers. */
    SIZE,

    /** The random seed; not read by this generator. */
    SEED,

    /** The op's debug name; not read by this generator. */
    NAME;

    /**
     * The keyword name.
     *
     * @return The lowercased enum name.
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    /**
     * The positional index, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  protected RandomCrop(PointsToSetVariable source) {
    super(source);
  }

  protected RandomCrop(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> sizePts =
        this.getArgumentPointsToSet(builder, Parameters.SIZE.getIndex(), Parameters.SIZE.getName());
    if (sizePts == null || sizePts.isEmpty()) return null;
    // A runtime-tensor size degrades to ⊤ rather than throwing (wala/ML#471).
    Set<List<Dimension<?>>> shapes = this.getShapesFromShapeArgument(builder, sizePts);
    return shapes == null || shapes.isEmpty() ? null : shapes;
  }

  /**
   * Reads the shape from the {@code size} argument rather than the parent's input pass-through.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The shapes per {@link #getDefaultShapes}.
   */
  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    return getDefaultShapes(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int valueVn =
        this.getArgumentValueNumber(
            builder, Parameters.VALUE.getIndex(), Parameters.VALUE.getName(), false);
    Set<DType> dtypes = this.getDTypes(builder, valueVn);
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SIZE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SIZE.getName();
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
