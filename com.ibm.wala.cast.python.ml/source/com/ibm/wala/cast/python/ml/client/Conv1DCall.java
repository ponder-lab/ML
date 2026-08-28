package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for a {@code tf.keras.layers.Conv1D} layer call (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>A one-dimensional convolution keeps the batch axis, rewrites the single temporal axis, and
 * sets the channel axis to the layer's {@code filters}. Unmodeled, it dropped the shape for the
 * rest of its chain, which is what kept a parallel-kernel sentence encoder rankless downstream of
 * its pooling and concatenation.
 *
 * <p>The arithmetic and the degradation rules are {@link ConvolutionCall}'s; only the rank differs.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Conv1D">tf.keras.layers.Conv1D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Conv1DCall extends ConvolutionCall {

  /** Rank of a one-dimensional convolution's input: batch, one temporal axis, and channels. */
  private static final int CONV1D_RANK = 3;

  /**
   * Constructs a {@code Conv1DCall} from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the call.
   */
  public Conv1DCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic method.
   */
  public Conv1DCall(CGNode node) {
    super(node);
  }

  @Override
  protected int getRank() {
    return CONV1D_RANK;
  }

  @Override
  protected String getLayerName() {
    return "Conv1D";
  }
}
