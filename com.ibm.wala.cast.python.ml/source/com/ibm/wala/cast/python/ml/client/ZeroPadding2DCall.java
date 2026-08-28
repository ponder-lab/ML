package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for the {@code __call__} on a {@code ZeroPadding2D} Keras layer instance. The output
 * keeps the batch and channel axes and grows each of the two spatial axes by the padding stored on
 * the layer instance (<a href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>Only a scalar {@code padding} resolves. Keras also accepts a pair of integers and a pair of
 * pairs, which name the two axes and the two sides separately; neither is proven single-valued by
 * the substrate, so those spatial extents degrade rather than being guessed at.
 *
 * <p>A {@link DynamicDim} spatial extent stays {@code Dynamic}: padding a feed-dependent axis
 * yields a feed-dependent axis, since the runtime shape still reports {@code None} there. An extent
 * that is fixed but uncomputed is {@link UnresolvedDim}, which carries no such {@code None}
 * evidence.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/ZeroPadding2D">tf.keras.layers.ZeroPadding2D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ZeroPadding2DCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(ZeroPadding2DCall.class.getName());

  /** Rank of a 2-D padding layer's input: batch, two spatial axes, and channels. */
  private static final int ZERO_PADDING_2D_RANK = 4;

  /** The constructor argument the model file stores on the layer instance. */
  private static final String PADDING_FIELD_NAME = "padding";

  /** Keras's default {@code padding}, one row or column added on each of the four sides. */
  private static final int DEFAULT_PADDING = 1;

  /**
   * Constructs a {@code ZeroPadding2DCall} from a caller-side {@link PointsToSetVariable} (the
   * result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code ZeroPadding2D} layer instance.
   */
  public ZeroPadding2DCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code ZeroPadding2DCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public ZeroPadding2DCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by keeping the batch and channel axes of each rank-4 input and
   * growing the two spatial axes by twice the padding, once per side.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, one per rank-4 input shape, or {@code null} if the input has no
   *     known shape, which is ⊤ rather than "not a tensor": the layer still returns a tensor.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 1, "inputs");
    if (inputPts == null || inputPts.isEmpty()) return null;
    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    Integer padding = this.resolvePadding(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only a rank-4 input is a 2-D padding layer's input. Anything else is not this operation
      // being applied as documented, so say nothing rather than invent a shape.
      if (inputShape.size() != ZERO_PADDING_2D_RANK) continue;

      List<Dimension<?>> outShape = new ArrayList<>(ZERO_PADDING_2D_RANK);
      outShape.add(inputShape.get(0));
      outShape.add(paddedExtent(inputShape.get(1), padding));
      outShape.add(paddedExtent(inputShape.get(2), padding));
      outShape.add(inputShape.get(3));
      ret.add(outShape);
    }

    if (ret.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + ZERO_PADDING_2D_RANK
                  + " input shape for 2-D padding call at: "
                  + describe(this.getNode()));
      return null;
    }

    return ret;
  }

  /**
   * Grows one spatial extent by the padding applied to each of its two sides.
   *
   * @param extent The input's extent on this axis.
   * @param padding The resolved per-side padding, or {@code null} when it did not resolve.
   * @return The output's extent on this axis.
   */
  private static Dimension<?> paddedExtent(Dimension<?> extent, Integer padding) {
    // Feed-dependence survives padding: the runtime shape reports None for a padded None axis.
    if (extent instanceof DynamicDim) return DynamicDim.INSTANCE;

    if (padding != null && extent instanceof NumericDim)
      return new NumericDim(((NumericDim) extent).value() + 2 * padding);

    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the per-side padding from the {@code padding} field the constructor summary stores on
   * the layer instance. An unsupplied argument takes Keras's default of one.
   *
   * @param builder The propagation call graph builder.
   * @return The padding, or {@code null} when it is not statically proven single-valued (a tuple
   *     argument, a non-constant, or several receiver instances that disagree).
   */
  private Integer resolvePadding(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return null;

    Set<Long> paddings = HashSetFactory.make();

    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return null;

      Set<Long> values =
          getPossibleLongValues(getInstanceFieldPointsToSet(builder, selfAsin, PADDING_FIELD_NAME));
      if (values == null) return null;
      paddings.addAll(values);
    }

    // An unsupplied or explicit-None `padding` takes the documented default.
    if (paddings.isEmpty() || (paddings.size() == 1 && paddings.contains(null)))
      return DEFAULT_PADDING;
    if (paddings.size() != 1) return null;

    int padding = paddings.iterator().next().intValue();
    return padding < 0 ? null : padding;
  }

  /**
   * Resolves the output dtypes by passing the {@code inputs} argument's dtypes through unchanged.
   *
   * @param builder The propagation call graph builder.
   * @return The set of dtypes observed on the input, or {@code {UNKNOWN\}} if none can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 1, "inputs");
    if (inputPts == null || inputPts.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = this.getDTypesOfValue(builder, inputPts);
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
