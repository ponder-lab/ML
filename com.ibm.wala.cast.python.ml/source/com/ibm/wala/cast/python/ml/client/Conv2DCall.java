package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for a {@code tf.keras.layers.Conv2D} layer call.
 *
 * <p>A convolution keeps the batch axis, rewrites the two spatial axes, and sets the channel axis
 * to the layer's {@code filters}. Without this, the call result carried no shape at all and every
 * value downstream of it lost its rank, which is a worse answer than a loose one: a specification
 * with no rank admits every argument. See wala/ML#820.
 *
 * <p>Recovering the rank is what unblocks the chain. A {@code Flatten} then {@code Dense} following
 * a convolution reaches the correct final shape from a rank-4 input whatever the spatial extents
 * are, because {@code Dense} rewrites only the last axis.
 *
 * <p>The spatial extents themselves are folded by {@link ConvolutionCall}, which documents when
 * they resolve and when they degrade.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D">Conv2D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Conv2DCall extends ConvolutionCall {

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
  protected int getRank() {
    return CONV2D_RANK;
  }

  @Override
  protected String getLayerName() {
    return "Conv2D";
  }
}
