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
 * Generator for the {@code __call__} on a {@code Conv2DTranspose} Keras layer instance. The output
 * keeps the batch axis, rewrites the two spatial axes upward from the strides, kernel size and
 * padding mode stored on the layer instance, and replaces the channel axis with the layer's filter
 * count (<a href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>The two axes are resolved independently of each other, which matters here more than it does
 * for pooling: {@code filters} is a required positional argument and is a literal at every call
 * site in ordinary generator code, so the channel axis usually resolves even where the spatial
 * arithmetic degrades. Reporting a known channel count alongside two unresolved spatial extents is
 * strictly more than the unknown rank this layer produced before.
 *
 * <p>Only a scalar {@code strides} and {@code kernel_size} resolve. Keras also accepts a pair for
 * each, naming the two spatial axes separately; neither is proven single-valued by the substrate,
 * so those extents degrade rather than being guessed at. {@code output_padding} is not modeled:
 * when it is supplied the spatial extents degrade, since it shifts them by an amount this generator
 * does not read.
 *
 * <p>A {@link DynamicDim} spatial extent stays {@code Dynamic}: upsampling a feed-dependent axis
 * yields a feed-dependent axis. An extent that is fixed but uncomputed is {@link UnresolvedDim},
 * which carries no {@code None} evidence either way.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Conv2DTranspose">tf.keras.layers.Conv2DTranspose</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Conv2DTransposeCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Conv2DTransposeCall.class.getName());

  /** Rank of a 2-D transposed convolution's input: batch, two spatial axes, and channels. */
  private static final int CONV_2D_TRANSPOSE_RANK = 4;

  /** The constructor argument the model file stores on the layer instance. */
  private static final String FILTERS_FIELD_NAME = "filters";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String KERNEL_SIZE_FIELD_NAME = "kernel_size";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String STRIDES_FIELD_NAME = "strides";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String PADDING_FIELD_NAME = "padding";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String OUTPUT_PADDING_FIELD_NAME = "output_padding";

  /** The padding mode whose output extent is the input extent scaled by the stride. */
  private static final String SAME_PADDING = "same";

  /** The padding mode whose output extent also spans the trailing kernel window. */
  private static final String VALID_PADDING = "valid";

  /** Keras's default {@code strides}, which leaves the spatial extents at their input size. */
  private static final int DEFAULT_STRIDE = 1;

  /**
   * A statically-resolved upsampling window: one kernel extent, one stride, and the padding mode,
   * all proven single-valued. Scalar constructor arguments only; a tuple-valued {@code kernel_size}
   * or {@code strides} does not resolve and the spatial extents degrade instead.
   */
  private static final class Window {
    final int kernelSize;
    final int stride;
    final boolean samePadding;

    Window(int kernelSize, int stride, boolean samePadding) {
      this.kernelSize = kernelSize;
      this.stride = stride;
      this.samePadding = samePadding;
    }
  }

  /**
   * Constructs a {@code Conv2DTransposeCall} from a caller-side {@link PointsToSetVariable} (the
   * result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code Conv2DTranspose} layer instance.
   */
  public Conv2DTransposeCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code Conv2DTransposeCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public Conv2DTransposeCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by keeping the batch axis of each rank-4 input, rewriting the two
   * spatial axes through the upsampling window, and replacing the channel axis with the filter
   * count.
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

    Window window = this.resolveWindow(builder);
    Dimension<?> channels = this.resolveFilters(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only a rank-4 input is a 2-D transposed convolution's input. Anything else is not this
      // operation being applied as documented, so say nothing rather than invent a shape.
      if (inputShape.size() != CONV_2D_TRANSPOSE_RANK) continue;

      List<Dimension<?>> outShape = new ArrayList<>(CONV_2D_TRANSPOSE_RANK);
      outShape.add(inputShape.get(0));
      outShape.add(upsampledExtent(inputShape.get(1), window));
      outShape.add(upsampledExtent(inputShape.get(2), window));
      outShape.add(channels);
      ret.add(outShape);
    }

    if (ret.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + CONV_2D_TRANSPOSE_RANK
                  + " input shape for 2-D transposed convolution call at: "
                  + describe(this.getNode()));
      return null;
    }

    return ret;
  }

  /**
   * Rewrites one spatial extent through the upsampling window.
   *
   * @param extent The input's extent on this axis.
   * @param window The resolved upsampling window, or {@code null} when it did not resolve.
   * @return The output's extent on this axis.
   */
  private static Dimension<?> upsampledExtent(Dimension<?> extent, Window window) {
    // Feed-dependence survives upsampling: the runtime shape reports None for a scaled None axis.
    if (extent instanceof DynamicDim) return DynamicDim.INSTANCE;

    if (window != null && extent instanceof NumericDim) {
      int size = ((NumericDim) extent).value();
      int upsampled =
          window.samePadding
              ? size * window.stride
              : (size - 1) * window.stride + window.kernelSize;
      if (upsampled > 0) return new NumericDim(upsampled);
    }

    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the output channel axis from the {@code filters} field the constructor summary stores
   * on the layer instance.
   *
   * @param builder The propagation call graph builder.
   * @return The filter count, or {@link UnresolvedDim} when it is not statically proven
   *     single-valued. It is never {@link DynamicDim}: a filter count is a fixed property of the
   *     layer, so the runtime shape does not report {@code None} on this axis.
   */
  private Dimension<?> resolveFilters(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return UnresolvedDim.INSTANCE;

    Set<Long> filters = HashSetFactory.make();

    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return UnresolvedDim.INSTANCE;

      Set<Long> values =
          getPossibleLongValues(getInstanceFieldPointsToSet(builder, selfAsin, FILTERS_FIELD_NAME));
      if (values == null) return UnresolvedDim.INSTANCE;
      filters.addAll(values);
    }

    if (filters.size() != 1 || filters.contains(null)) return UnresolvedDim.INSTANCE;

    int count = filters.iterator().next().intValue();
    return count > 0 ? new NumericDim(count) : UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the upsampling window from the {@code kernel_size}, {@code strides} and {@code
   * padding} fields the constructor summary stores on the layer instance. An unsupplied {@code
   * strides} takes Keras's default of one, and an unsupplied {@code padding} its default of {@code
   * "valid"}.
   *
   * @param builder The propagation call graph builder.
   * @return The window, or {@code null} when any of its parts is not statically proven
   *     single-valued (a tuple argument, a non-constant, a supplied {@code output_padding}, or
   *     several receiver instances that disagree).
   */
  private Window resolveWindow(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return null;

    Set<Long> kernelSizes = HashSetFactory.make();
    Set<Long> strides = HashSetFactory.make();
    Set<Object> paddings = HashSetFactory.make();

    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return null;

      // `output_padding` shifts the spatial extents by an amount this generator does not read, so
      // any supplied value means the arithmetic below no longer describes the output.
      Set<Long> outputPadding =
          getPossibleLongValues(
              getInstanceFieldPointsToSet(builder, selfAsin, OUTPUT_PADDING_FIELD_NAME));
      if (outputPadding == null) return null;
      if (!outputPadding.isEmpty() && !outputPadding.equals(java.util.Collections.singleton(null)))
        return null;

      Set<Long> kernelValues =
          getPossibleLongValues(
              getInstanceFieldPointsToSet(builder, selfAsin, KERNEL_SIZE_FIELD_NAME));
      if (kernelValues == null) return null;
      kernelSizes.addAll(kernelValues);

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

    if (kernelSizes.size() != 1 || kernelSizes.contains(null)) return null;
    int kernelSize = kernelSizes.iterator().next().intValue();

    // An unsupplied or explicit-None `strides` leaves the spatial extents at their input size.
    int stride;
    if (strides.isEmpty() || (strides.size() == 1 && strides.contains(null)))
      stride = DEFAULT_STRIDE;
    else if (strides.size() == 1) stride = strides.iterator().next().intValue();
    else return null;

    boolean samePadding;
    if (paddings.isEmpty()) samePadding = false;
    else if (paddings.size() == 1 && VALID_PADDING.equals(paddings.iterator().next()))
      samePadding = false;
    else if (paddings.size() == 1 && SAME_PADDING.equals(paddings.iterator().next()))
      samePadding = true;
    else return null;

    if (kernelSize <= 0 || stride <= 0) return null;
    return new Window(kernelSize, stride, samePadding);
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
