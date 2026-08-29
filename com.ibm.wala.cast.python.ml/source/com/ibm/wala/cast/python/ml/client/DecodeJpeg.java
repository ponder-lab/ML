package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
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
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for {@code tf.io.decode_jpeg(contents, channels=0, ...)}: rank 3 with every extent
 * dynamic, by the API contract. TensorFlow's own static shape for a decoded JPEG is {@code (None,
 * None, None)} — height, width, and channel count all come from the encoded bytes — so the rank is
 * certain while no extent is, the {@link DynamicDim} criterion exactly (each axis reports {@code
 * None} at trace time). The concrete sizes live in the image files, not the program, so unknown
 * rank here loses a rank the contract proves and a concrete extent would be fabricated
 * (wala/ML#853).
 *
 * <p>The one refinement the arguments allow: a literal nonzero {@code channels} pins the channel
 * axis (TensorFlow's static shape then reports it, e.g. {@code (None, None, 3)}); {@code 0} or an
 * unresolved value keeps it dynamic.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/io/decode_jpeg">
 *     tf.io.decode_jpeg</a>.
 */
public class DecodeJpeg extends TensorTypeAllocator {

  protected enum Parameters {
    CONTENTS,
    CHANNELS;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal() + 1; // After the implicit receiver.
    }
  }

  public DecodeJpeg(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a manual-node generator for the tensor allocated in {@code decode_jpeg.do()}, used
   * by {@link TensorGenerator#createManualGenerator(CGNode, PropagationCallGraphBuilder)} when the
   * result reaches a consumer through producer delegation.
   *
   * @param node The {@code decode_jpeg.do()} call-graph node.
   */
  public DecodeJpeg(CGNode node) {
    super(node);
  }

  /**
   * Returns the rank-3 contract shape, with the channel axis pinned when a literal nonzero {@code
   * channels} argument proves it and dynamic otherwise.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A single rank-3 shape: two dynamic spatial axes and the channel axis.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Dimension<?> channels = DynamicDim.INSTANCE;

    OrdinalSet<InstanceKey> channelsPts =
        this.getArgumentPointsToSet(
            builder, Parameters.CHANNELS.getIndex(), Parameters.CHANNELS.getName());
    if (channelsPts != null && !channelsPts.isEmpty()) {
      Set<Long> values = getPossibleLongValues(channelsPts);
      if (values != null && values.size() == 1) {
        Long value = values.iterator().next();
        // `channels=0` means "use the JPEG's own channel count": dynamic, not zero.
        if (value != null && value > 0) channels = new NumericDim(value.intValue());
      }
    }

    List<Dimension<?>> shape = new ArrayList<>(3);
    shape.add(DynamicDim.INSTANCE);
    shape.add(DynamicDim.INSTANCE);
    shape.add(channels);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(shape);
    return ret;
  }

  /**
   * Returns the {@code uint8} dtype the API contract fixes.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link DType#UINT8}, alone.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UINT8);
  }

  /** No shape argument: the shape is the contract's, refined only by {@code channels}. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /** No dtype argument: the dtype is the contract's. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
