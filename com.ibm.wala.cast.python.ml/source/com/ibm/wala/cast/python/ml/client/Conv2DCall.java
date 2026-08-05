package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for a {@code tf.keras.layers.Conv2D} layer call.
 *
 * <p>A convolution keeps the batch axis, rewrites the two spatial axes, and sets the channel axis
 * to the layer's {@code filters}. Without this, the call result carried no shape at all and every
 * value downstream of it lost its rank, which is a worse answer than a loose one: a specification
 * with no rank admits every argument. See wala/ML#820.
 *
 * <p>The spatial extents are reported as {@link UnresolvedDim}. They are fixed integers at run
 * time, determined by the input extent together with {@code kernel_size}, {@code strides}, {@code
 * padding} and {@code dilation_rate}, and folding that arithmetic is separate work. Per the
 * dimension conventions an extent the analysis cannot compute but which is not feed-dependent is
 * {@code Unresolved} rather than {@code Dynamic}: nothing here carries evidence that the runtime
 * shape would report {@code None}.
 *
 * <p>Recovering the rank is what unblocks the chain. A {@code Flatten} then {@code Dense} following
 * a convolution reaches the correct final shape from a rank-4 input whatever the spatial extents
 * are, because {@code Dense} rewrites only the last axis.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D">Conv2D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Conv2DCall extends DenseCall {

  private static final Logger LOGGER = Logger.getLogger(Conv2DCall.class.getName());

  /** The constructor argument the model file stores on the layer instance. */
  private static final String FILTERS_FIELD_NAME = "filters";

  /** Rank of a two-dimensional convolution's input: batch, two spatial axes, and channels. */
  private static final int CONV2D_RANK = 4;

  public Conv2DCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Conv2DCall(CGNode node) {
    super(node);
  }

  @Override
  protected String getUnitsFieldName() {
    return FILTERS_FIELD_NAME;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = this.getInputShapes(builder);
    if (inputShapes == null) return null;

    Set<Long> filterValues = this.getPossibleUnits(builder);

    Set<List<Dimension<?>>> outputShapes = new HashSet<>();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only a rank-4 input is a two-dimensional convolution's input. Anything else is not this
      // operation being applied as documented, so say nothing rather than invent a shape.
      if (inputShape.size() != CONV2D_RANK) continue;

      for (Long filters :
          filterValues.isEmpty() ? Collections.<Long>singleton(null) : filterValues) {
        List<Dimension<?>> outShape = new ArrayList<>(CONV2D_RANK);
        outShape.add(inputShape.get(0));
        outShape.add(UnresolvedDim.INSTANCE);
        outShape.add(UnresolvedDim.INSTANCE);
        outShape.add(filters == null ? UnresolvedDim.INSTANCE : new NumericDim(filters.intValue()));
        outputShapes.add(outShape);
      }
    }

    if (outputShapes.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + CONV2D_RANK
                  + " input shape for Conv2D call at: "
                  + describe(this.getNode()));
      return null;
    }

    return outputShapes;
  }
}
