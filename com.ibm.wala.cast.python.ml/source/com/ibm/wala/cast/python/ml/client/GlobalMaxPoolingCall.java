package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for the {@code __call__} on a global-maximum-pooling Keras layer instance ({@code
 * GlobalMaxPooling1D} and {@code GlobalMaxPooling2D}, with their alias spellings). The interior
 * axes are pooled away: the output keeps the batch and feature axes, so a rank-3 input {@code (B,
 * steps, features)} yields {@code (B, features)} and a rank-4 input {@code (B, rows, cols,
 * channels)} yields {@code (B, channels)}; dtype passes through unchanged (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>). Mirrors {@link
 * GlobalAveragePooling1DCall}, parameterized by the input rank the layer class fixes.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/GlobalMaxPool1D">tf.keras.layers.GlobalMaxPooling1D</a>
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/GlobalMaxPool2D">tf.keras.layers.GlobalMaxPooling2D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class GlobalMaxPoolingCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(GlobalMaxPoolingCall.class.getName());

  /** Rank of a 1-D global pooling layer's input: batch, steps, and features. */
  public static final int INPUT_RANK_1D = 3;

  /** Rank of a 2-D global pooling layer's input: batch, two spatial axes, and channels. */
  public static final int INPUT_RANK_2D = 4;

  /** The input rank the layer class fixes; only inputs of this rank produce an output shape. */
  private final int inputRank;

  /**
   * Constructs a {@code GlobalMaxPoolingCall} from a caller-side {@link PointsToSetVariable} (the
   * result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a global-maximum-pooling layer instance.
   * @param inputRank The input rank the layer class fixes ({@link #INPUT_RANK_1D} or {@link
   *     #INPUT_RANK_2D}).
   */
  public GlobalMaxPoolingCall(PointsToSetVariable source, int inputRank) {
    super(source);
    this.inputRank = inputRank;
  }

  /**
   * Constructs a {@code GlobalMaxPoolingCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   * @param inputRank The input rank the layer class fixes ({@link #INPUT_RANK_1D} or {@link
   *     #INPUT_RANK_2D}).
   */
  public GlobalMaxPoolingCall(CGNode node, int inputRank) {
    super(node);
    this.inputRank = inputRank;
  }

  /**
   * Resolves the output shapes by keeping the first (batch) and last (feature) axes of each input
   * whose rank matches the layer class's.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, one per matching input shape, or {@code null} if the input has
   *     no known shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The SSA-chain fallback covers an input with dataflow state but no points-to evidence, e.g.
    // a destructured tuple dataset element (wala/ML#855).
    Set<List<Dimension<?>>> inputShapes = this.getArgumentShapesWithFallback(builder, 1, "inputs");
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only an input of the class's rank is this operation being applied as documented, so say
      // nothing rather than invent a shape.
      if (inputShape.size() != this.inputRank) continue;

      List<Dimension<?>> outShape = new ArrayList<>(2);
      outShape.add(inputShape.get(0));
      outShape.add(inputShape.get(inputShape.size() - 1));
      ret.add(outShape);
    }

    if (ret.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + this.inputRank
                  + " input shape for global pooling call at: "
                  + describe(this.getNode()));
      return null;
    }

    return ret;
  }

  /**
   * Resolves the output dtypes by passing the {@code inputs} argument's dtypes through unchanged.
   *
   * @param builder The propagation call graph builder.
   * @return The set of dtypes observed on the input, or {@code {UNKNOWN\}} if none can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // The dtype twin of the shape fallback (wala/ML#855).
    Set<DType> dtypes = this.getArgumentDTypesWithFallback(builder, 1, "inputs");
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; the call takes no explicit shape parameter.
   */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; the call takes no explicit shape parameter.
   */
  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; the call takes no explicit dtype parameter.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; the call takes no explicit dtype parameter.
   */
  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
