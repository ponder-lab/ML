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
 * Generator for the {@code __call__} on a 2-D window-pooling Keras layer instance ({@code
 * MaxPool2D} and {@code AveragePooling2D}, with their alias spellings). The output keeps the batch
 * and channel axes and rewrites the two spatial axes from the pool size, strides, and padding
 * stored on the layer instance (<a href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>This is deliberately one generator rather than one per class: maximum and average pooling
 * differ in what they compute and not in how they transform the type.
 *
 * <p>A spatial extent is folded to a number only when both the input extent and the whole pooling
 * window resolve statically; a fixed extent the analysis cannot compute is {@link UnresolvedDim}
 * rather than {@link DynamicDim}, since nothing here carries evidence that the runtime shape would
 * report {@code None}. A {@link DynamicDim} input extent stays {@code Dynamic}: pooling a
 * feed-dependent axis yields a feed-dependent axis.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/MaxPool2D">tf.keras.layers.MaxPool2D</a>
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/AveragePooling2D">tf.keras.layers.AveragePooling2D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Pooling2DCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Pooling2DCall.class.getName());

  /** Rank of a 2-D pooling layer's input: batch, two spatial axes, and channels. */
  private static final int POOLING_2D_RANK = 4;

  /** The constructor argument the model file stores on the layer instance. */
  private static final String POOL_SIZE_FIELD_NAME = "pool_size";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String STRIDES_FIELD_NAME = "strides";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String PADDING_FIELD_NAME = "padding";

  /** The padding mode that drops incomplete windows: the extent is ⌊(size − pool) / stride⌋ + 1. */
  private static final String VALID_PADDING = "valid";

  /** The padding mode that pads incomplete windows: the extent is ⌈size / stride⌉. */
  private static final String SAME_PADDING = "same";

  /**
   * A statically-resolved pooling window: one pool extent, one stride, and the padding mode, all
   * proven single-valued. Scalar constructor arguments only; a tuple-valued {@code pool_size} or
   * {@code strides} does not resolve and the spatial extents degrade instead.
   */
  private static final class Window {
    final int poolSize;
    final int stride;
    final boolean samePadding;

    Window(int poolSize, int stride, boolean samePadding) {
      this.poolSize = poolSize;
      this.stride = stride;
      this.samePadding = samePadding;
    }
  }

  /**
   * Constructs a {@code Pooling2DCall} from a caller-side {@link PointsToSetVariable} (the result
   * of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a 2-D pooling layer instance.
   */
  public Pooling2DCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code Pooling2DCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public Pooling2DCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by keeping the batch and channel axes of each rank-4 input and
   * rewriting the two spatial axes from the layer's pooling window.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, one per rank-4 input shape, or {@code null} if the input has no
   *     known shape, which is ⊤ rather than "not a tensor": the layer still returns a tensor.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The SSA-chain fallback covers an input with dataflow state but no points-to evidence, e.g.
    // a destructured tuple dataset element (wala/ML#855).
    Set<List<Dimension<?>>> inputShapes = this.getArgumentShapesWithFallback(builder, 1, "inputs");
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    Window window = this.resolveWindow(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only a rank-4 input is a 2-D pooling layer's input. Anything else is not this operation
      // being applied as documented, so say nothing rather than invent a shape.
      if (inputShape.size() != POOLING_2D_RANK) continue;

      List<Dimension<?>> outShape = new ArrayList<>(POOLING_2D_RANK);
      outShape.add(inputShape.get(0));
      outShape.add(pooledExtent(inputShape.get(1), window));
      outShape.add(pooledExtent(inputShape.get(2), window));
      outShape.add(inputShape.get(3));
      ret.add(outShape);
    }

    if (ret.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + POOLING_2D_RANK
                  + " input shape for 2-D pooling call at: "
                  + describe(this.getNode()));
      return null;
    }

    return ret;
  }

  /**
   * Rewrites one spatial extent through the pooling window.
   *
   * @param extent The input's extent on this axis.
   * @param window The resolved pooling window, or {@code null} when it did not resolve.
   * @return The output's extent on this axis.
   */
  private static Dimension<?> pooledExtent(Dimension<?> extent, Window window) {
    // Feed-dependence survives pooling: the runtime shape reports None for a pooled None axis.
    if (extent instanceof DynamicDim) return DynamicDim.INSTANCE;

    if (window != null && extent instanceof NumericDim) {
      int size = ((NumericDim) extent).value();
      int pooled =
          window.samePadding
              ? Math.floorDiv(size + window.stride - 1, window.stride)
              : Math.floorDiv(size - window.poolSize, window.stride) + 1;
      if (pooled > 0) return new NumericDim(pooled);
    }

    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the pooling window from the {@code pool_size}, {@code strides}, and {@code padding}
   * fields the constructor summary stores on the layer instance. Unsupplied arguments take Keras's
   * defaults: {@code strides} of {@code None} follows the pool size, and padding defaults to {@code
   * "valid"}.
   *
   * @param builder The propagation call graph builder.
   * @return The window, or {@code null} when any of its parts is not statically proven
   *     single-valued (a tuple argument, a non-constant, or several receiver instances that
   *     disagree).
   */
  private Window resolveWindow(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return null;

    Set<Long> poolSizes = HashSetFactory.make();
    Set<Long> strides = HashSetFactory.make();
    Set<Object> paddings = HashSetFactory.make();

    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return null;

      Set<Long> poolValues =
          getPossibleLongValues(
              getInstanceFieldPointsToSet(builder, selfAsin, POOL_SIZE_FIELD_NAME));
      if (poolValues == null) return null;
      poolSizes.addAll(poolValues);

      Set<Long> strideValues =
          getPossibleLongValues(getInstanceFieldPointsToSet(builder, selfAsin, STRIDES_FIELD_NAME));
      if (strideValues == null) return null;
      strides.addAll(strideValues);

      Set<Object> paddingValues =
          getConstantValues(
              getInstanceFieldPointsToSet(builder, selfAsin, PADDING_FIELD_NAME), true);
      if (paddingValues == null) return null;
      // Keras normalizes the padding mode case-insensitively (`normalize_padding` lower-cases
      // it), and real code writes `'SAME'` as often as `'same'`; normalize at collection so the
      // spellings agree and the comparison below stays exact.
      for (Object paddingValue : paddingValues)
        paddings.add(
            paddingValue instanceof String
                ? ((String) paddingValue).toLowerCase(java.util.Locale.ROOT)
                : paddingValue);
    }

    if (poolSizes.size() != 1 || poolSizes.contains(null)) return null;
    int poolSize = poolSizes.iterator().next().intValue();

    // An unsupplied or explicit-None `strides` follows the pool size.
    int stride;
    if (strides.isEmpty() || (strides.size() == 1 && strides.contains(null))) stride = poolSize;
    else if (strides.size() == 1) stride = strides.iterator().next().intValue();
    else return null;

    boolean samePadding;
    if (paddings.isEmpty()) samePadding = false;
    else if (paddings.size() == 1 && VALID_PADDING.equals(paddings.iterator().next()))
      samePadding = false;
    else if (paddings.size() == 1 && SAME_PADDING.equals(paddings.iterator().next()))
      samePadding = true;
    else return null;

    if (poolSize <= 0 || stride <= 0) return null;
    return new Window(poolSize, stride, samePadding);
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
