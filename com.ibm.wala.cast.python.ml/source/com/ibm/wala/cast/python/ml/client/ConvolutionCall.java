package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Common behavior for a forward convolution layer call ({@code Conv1D}, {@code Conv2D}). The output
 * keeps the batch axis, rewrites each spatial axis through the convolution window, and sets the
 * channel axis to the layer's {@code filters}.
 *
 * <p>Folding the spatial arithmetic was left as separate work when the rank was first recovered
 * (wala/ML#820); this class does it (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>). It matters beyond its own layer: a
 * convolution that reports {@code Unresolved} spatial extents discards whatever a preceding layer
 * resolved, so an unfolded convolution downstream of a folded padding or pooling layer throws that
 * work away.
 *
 * <p>The spatial and channel axes resolve independently. {@code filters} is a required positional
 * argument and a literal at essentially every ordinary call site, so the channel axis usually
 * resolves even where the window does not.
 *
 * <p>Only scalar window arguments resolve. Keras also accepts a tuple for {@code kernel_size},
 * {@code strides} and {@code dilation_rate}, naming each spatial axis separately, and none of those
 * is proven single-valued by the substrate, so those extents degrade instead of being guessed at.
 *
 * <p>A {@link DynamicDim} spatial extent stays {@code Dynamic}: convolving a feed-dependent axis
 * yields a feed-dependent axis, since the runtime shape still reports {@code None} there. An extent
 * that is fixed but uncomputed is {@link UnresolvedDim}, which carries no such evidence either way.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class ConvolutionCall extends DenseCall {

  private static final Logger LOGGER = Logger.getLogger(ConvolutionCall.class.getName());

  /** The constructor argument the model file stores on the layer instance. */
  protected static final String FILTERS_FIELD_NAME = "filters";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String KERNEL_SIZE_FIELD_NAME = "kernel_size";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String STRIDES_FIELD_NAME = "strides";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String PADDING_FIELD_NAME = "padding";

  /** The constructor argument the model file stores on the layer instance. */
  private static final String DILATION_RATE_FIELD_NAME = "dilation_rate";

  /** The padding mode that drops incomplete windows: the extent shrinks by the window's span. */
  private static final String VALID_PADDING = "valid";

  /** The padding mode that pads incomplete windows: the extent is ⌈size / stride⌉. */
  private static final String SAME_PADDING = "same";

  /** Keras's default {@code strides}, which advances the window one step at a time. */
  private static final int DEFAULT_STRIDE = 1;

  /** Keras's default {@code dilation_rate}, which leaves the kernel's span equal to its size. */
  private static final int DEFAULT_DILATION = 1;

  /**
   * A statically-resolved convolution window: one kernel extent, one stride, one dilation rate and
   * the padding mode, all proven single-valued.
   */
  private static final class Window {
    final int kernelSize;
    final int stride;
    final int dilation;
    final boolean samePadding;

    Window(int kernelSize, int stride, int dilation, boolean samePadding) {
      this.kernelSize = kernelSize;
      this.stride = stride;
      this.dilation = dilation;
      this.samePadding = samePadding;
    }

    /** The kernel's span once dilation spreads it, which is what the extent actually loses. */
    int span() {
      return this.dilation * (this.kernelSize - 1) + 1;
    }
  }

  /**
   * Constructs a convolution call generator from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the call.
   */
  protected ConvolutionCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a convolution call generator anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic method.
   */
  protected ConvolutionCall(CGNode node) {
    super(node);
  }

  /**
   * @return The rank this convolution's input must have: batch, its spatial axes, and channels.
   */
  protected abstract int getRank();

  /**
   * @return The layer's simple name, for diagnostics.
   */
  protected abstract String getLayerName();

  @Override
  protected String getUnitsFieldName() {
    return FILTERS_FIELD_NAME;
  }

  /**
   * Resolves the output shapes by keeping the batch axis of each input of the expected rank,
   * rewriting every spatial axis through the convolution window, and setting the channel axis to
   * the filter count.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, or {@code null} if the input has no known shape, which is ⊤
   *     rather than "not a tensor": the layer still returns a tensor.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = this.getInputShapes(builder);
    if (inputShapes == null) return null;

    Set<Long> filterValues = this.getPossibleUnits(builder);
    Window window = this.resolveWindow(builder);
    int rank = this.getRank();

    Set<List<Dimension<?>>> outputShapes = new HashSet<>();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only an input of the documented rank is this convolution's input. Anything else is not this
      // operation being applied as documented, so say nothing rather than invent a shape.
      if (inputShape.size() != rank) continue;

      for (Long filters :
          filterValues.isEmpty() ? Collections.<Long>singleton(null) : filterValues) {
        List<Dimension<?>> outShape = new ArrayList<>(rank);
        outShape.add(inputShape.get(0));

        // Every axis between the batch axis and the channel axis is spatial.
        for (int axis = 1; axis < rank - 1; axis++)
          outShape.add(convolvedExtent(inputShape.get(axis), window));

        outShape.add(filters == null ? UnresolvedDim.INSTANCE : new NumericDim(filters.intValue()));
        outputShapes.add(outShape);
      }
    }

    if (outputShapes.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + this.getRank()
                  + " input shape for "
                  + this.getLayerName()
                  + " call at: "
                  + describe(this.getNode()));
      return null;
    }

    return outputShapes;
  }

  /**
   * Rewrites one spatial extent through the convolution window.
   *
   * @param extent The input's extent on this axis.
   * @param window The resolved window, or {@code null} when it did not resolve.
   * @return The output's extent on this axis.
   */
  private static Dimension<?> convolvedExtent(Dimension<?> extent, Window window) {
    // Feed-dependence survives convolution: the runtime shape reports None for a convolved None
    // axis.
    if (extent instanceof DynamicDim) return DynamicDim.INSTANCE;

    if (window != null && extent instanceof NumericDim) {
      int size = ((NumericDim) extent).value();
      int convolved =
          window.samePadding
              ? Math.floorDiv(size + window.stride - 1, window.stride)
              : Math.floorDiv(size - window.span(), window.stride) + 1;
      if (convolved > 0) return new NumericDim(convolved);
    }

    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the convolution window from the fields the constructor summary stores on the layer
   * instance. Unsupplied arguments take Keras's defaults: a stride and a dilation rate of one, and
   * {@code "valid"} padding.
   *
   * @param builder The propagation call graph builder.
   * @return The window, or {@code null} when any part is not statically proven single-valued (a
   *     tuple argument, a non-constant, or several receiver instances that disagree).
   */
  private Window resolveWindow(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());
    if (selfPts == null || selfPts.isEmpty()) return null;

    Set<Long> kernelSizes = HashSetFactory.make();
    Set<Long> strides = HashSetFactory.make();
    Set<Long> dilations = HashSetFactory.make();
    Set<Object> paddings = HashSetFactory.make();

    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return null;

      Set<Long> kernelValues =
          getPossibleLongValues(
              getInstanceFieldPointsToSet(builder, selfAsin, KERNEL_SIZE_FIELD_NAME));
      if (kernelValues == null) return null;
      kernelSizes.addAll(kernelValues);

      Set<Long> strideValues =
          getPossibleLongValues(getInstanceFieldPointsToSet(builder, selfAsin, STRIDES_FIELD_NAME));
      if (strideValues == null) return null;
      strides.addAll(strideValues);

      Set<Long> dilationValues =
          getPossibleLongValues(
              getInstanceFieldPointsToSet(builder, selfAsin, DILATION_RATE_FIELD_NAME));
      if (dilationValues == null) return null;
      dilations.addAll(dilationValues);

      Set<Object> paddingValues =
          getConstantValues(
              getInstanceFieldPointsToSet(builder, selfAsin, PADDING_FIELD_NAME), true);
      if (paddingValues == null) return null;
      paddings.addAll(paddingValues);
    }

    if (kernelSizes.size() != 1 || kernelSizes.contains(null)) return null;
    int kernelSize = kernelSizes.iterator().next().intValue();

    Integer stride = single(strides, DEFAULT_STRIDE);
    if (stride == null) return null;

    Integer dilation = single(dilations, DEFAULT_DILATION);
    if (dilation == null) return null;

    boolean samePadding;
    if (paddings.isEmpty()) samePadding = false;
    else if (paddings.size() == 1 && VALID_PADDING.equals(paddings.iterator().next()))
      samePadding = false;
    else if (paddings.size() == 1 && SAME_PADDING.equals(paddings.iterator().next()))
      samePadding = true;
    else return null;

    if (kernelSize <= 0 || stride <= 0 || dilation <= 0) return null;
    return new Window(kernelSize, stride, dilation, samePadding);
  }

  /**
   * Reduces a field's observed values to the single one it must carry, treating an unsupplied or
   * explicitly-{@code None} argument as the documented default.
   *
   * @param values The values observed on the field.
   * @param defaultValue Keras's default for the argument.
   * @return The single value, or {@code null} when the field does not resolve.
   */
  private static Integer single(Set<Long> values, int defaultValue) {
    if (values.isEmpty() || (values.size() == 1 && values.contains(null))) return defaultValue;
    if (values.size() != 1) return null;
    return values.iterator().next().intValue();
  }
}
