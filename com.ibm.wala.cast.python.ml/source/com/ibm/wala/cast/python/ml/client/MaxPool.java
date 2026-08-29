package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
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
import java.util.logging.Logger;

/**
 * A generator for {@code tf.nn.max_pool(input, ksize, strides, padding, ...)}: the batch and
 * channel axes pass through, and each spatial axis folds through the pooling window, mirroring the
 * 2-D pooling layer's arithmetic. All three window arguments are required by the API, so there are
 * no defaults to assume; a window that does not resolve to single scalar values degrades the
 * spatial axes instead of guessing (wala/ML#857 replaced a hard-coded stride-2 halving that ignored
 * the arguments entirely).
 *
 * <p>Only a rank-4 input transforms; any other rank is not this operation applied as documented, so
 * nothing is said rather than a shape invented. A {@link DynamicDim} spatial extent stays {@code
 * Dynamic}: pooling a feed-dependent axis yields a feed-dependent axis.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/nn/max_pool">
 *     tf.nn.max_pool</a>.
 */
public class MaxPool extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(MaxPool.class.getName());

  /** The rank the documented 4-D form transforms: batch, two spatial axes, channels. */
  private static final int POOLED_RANK = 4;

  /** The padding mode that drops incomplete windows. */
  private static final String VALID_PADDING = "valid";

  /** The padding mode that pads incomplete windows. */
  private static final String SAME_PADDING = "same";

  /**
   * Parameter positions and names for the call's arguments. Function-form positions are zero-based
   * over the real arguments, the {@code Cast} convention: the argument-resolution machinery maps
   * position 0 to the first argument after the callee.
   */
  protected enum Parameters {
    INPUT,
    KSIZE,
    STRIDES,
    PADDING;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /** A statically-resolved pooling window: one kernel extent, one stride, and the padding mode. */
  private static final class Window {
    final int ksize;
    final int stride;
    final boolean samePadding;

    Window(int ksize, int stride, boolean samePadding) {
      this.ksize = ksize;
      this.stride = stride;
      this.samePadding = samePadding;
    }
  }

  public MaxPool(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public MaxPool(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The SSA-chain fallback covers an input with dataflow state but no points-to evidence, e.g.
    // a destructured tuple dataset element (wala/ML#855).
    Set<List<Dimension<?>>> inputShapes =
        this.getArgumentShapesWithFallback(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    Window window = this.resolveWindow(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.size() != POOLED_RANK) continue;

      List<Dimension<?>> outShape = new ArrayList<>(POOLED_RANK);
      outShape.add(inputShape.get(0));
      outShape.add(pooledExtent(inputShape.get(1), window));
      outShape.add(pooledExtent(inputShape.get(2), window));
      outShape.add(inputShape.get(3));
      ret.add(outShape);
    }

    if (ret.isEmpty()) {
      LOGGER.fine(() -> "No rank-" + POOLED_RANK + " input shape for max_pool call.");
      return null;
    }

    return ret;
  }

  /**
   * Rewrites one spatial extent through the pooling window, mirroring {@code
   * Pooling2DCall.pooledExtent}: same padding is ⌈size / stride⌉ and valid padding is ⌊(size −
   * ksize) / stride⌋ + 1.
   *
   * @param extent The input's extent on this axis.
   * @param window The resolved window, or {@code null} when it did not resolve.
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
              : Math.floorDiv(size - window.ksize, window.stride) + 1;
      if (pooled > 0) return new NumericDim(pooled);
    }

    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the pooling window from the call's own {@code ksize}, {@code strides}, and {@code
   * padding} arguments, all of which the API requires. Only scalar window arguments resolve: a
   * list-valued {@code ksize} or {@code strides} names each axis separately and is not proven
   * single-valued, so those extents degrade instead of being guessed at. The padding mode is
   * normalized case-insensitively at collection, matching the runtime's own handling.
   *
   * @param builder The propagation call graph builder.
   * @return The window, or {@code null} when any part is not statically proven single-valued.
   */
  private Window resolveWindow(PropagationCallGraphBuilder builder) {
    Integer ksize = singleValue(builder, Parameters.KSIZE.getIndex(), Parameters.KSIZE.getName());
    if (ksize == null || ksize <= 0) return null;

    Integer stride =
        singleValue(builder, Parameters.STRIDES.getIndex(), Parameters.STRIDES.getName());
    if (stride == null || stride <= 0) return null;

    OrdinalSet<InstanceKey> paddingPts =
        this.getArgumentPointsToSet(
            builder, Parameters.PADDING.getIndex(), Parameters.PADDING.getName());
    Set<Object> paddings = getConstantValues(paddingPts, true);
    if (paddings == null || paddings.size() != 1) return null;
    Object padding = paddings.iterator().next();
    if (!(padding instanceof String)) return null;
    String mode = ((String) padding).toLowerCase(Locale.ROOT);

    if (VALID_PADDING.equals(mode)) return new Window(ksize, stride, false);
    if (SAME_PADDING.equals(mode)) return new Window(ksize, stride, true);
    return null;
  }

  /**
   * Resolves one window argument to the single scalar it must carry, or {@code null} when it does
   * not resolve to exactly one integer.
   *
   * @param builder The propagation call graph builder.
   * @param index The argument's positional index.
   * @param name The argument's keyword name.
   * @return The single value, or {@code null}.
   */
  private Integer singleValue(PropagationCallGraphBuilder builder, int index, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, index, name);
    Set<Long> values = getPossibleLongValues(pts);
    if (values == null || values.size() != 1) return null;
    Long value = values.iterator().next();
    return value == null ? null : value.intValue();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Pooling preserves the input dtype; the dtype twin of the shape fallback (wala/ML#855).
    Set<DType> dtypes =
        this.getArgumentDTypesWithFallback(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
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
