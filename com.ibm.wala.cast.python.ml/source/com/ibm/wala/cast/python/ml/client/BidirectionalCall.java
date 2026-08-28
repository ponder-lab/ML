package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
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
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Generator for the {@code __call__} on a {@code tf.keras.layers.Bidirectional} instance, the
 * wrapper that runs its wrapped recurrent layer over the input in both directions. The wrapped
 * layer's {@code units} and {@code return_sequences} govern the output exactly as they do for the
 * layer alone, read through the {@code layer} field the constructor summary stores on the wrapper;
 * {@code merge_mode} decides the output's width: Keras's default of {@code concat} doubles the
 * wrapped width, while {@code sum}, {@code mul}, and {@code ave} keep it (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>A {@code merge_mode} of {@code None} makes the call return a list of two tensors rather than
 * one, which this generator does not model; it degrades to an unresolved width rather than claiming
 * one.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Bidirectional">tf.keras.layers.Bidirectional</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class BidirectionalCall extends RecurrentLayerCall {

  /** The constructor argument the model file stores on the wrapper instance. */
  private static final String LAYER_FIELD_NAME = "layer";

  /** The constructor argument the model file stores on the wrapper instance. */
  private static final String MERGE_MODE_FIELD_NAME = "merge_mode";

  /** The merge mode that concatenates the two directions, doubling the width; Keras's default. */
  private static final String CONCAT_MERGE_MODE = "concat";

  /** The merge modes that combine the two directions element-wise, keeping the width. */
  private static final Set<String> WIDTH_PRESERVING_MERGE_MODES = Set.of("sum", "mul", "ave");

  /**
   * Constructs a {@code BidirectionalCall} from a caller-side {@link PointsToSetVariable} (the
   * result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code Bidirectional} instance.
   */
  public BidirectionalCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code BidirectionalCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public BidirectionalCall(CGNode node) {
    super(node);
  }

  /**
   * The wrapped recurrent layer's instances, read through the wrapper's {@code layer} field: they
   * are what carry {@code units} and {@code return_sequences} here, not the wrapper itself.
   *
   * @param builder The propagation call graph builder.
   * @return The allocation sites of the wrapped layer instances; possibly empty.
   */
  @Override
  protected Collection<AllocationSiteInNode> getParameterCarriers(
      PropagationCallGraphBuilder builder) {
    List<AllocationSiteInNode> ret = new ArrayList<>();
    for (AllocationSiteInNode wrapper : super.getParameterCarriers(builder)) {
      OrdinalSet<InstanceKey> layerPts =
          getInstanceFieldPointsToSet(builder, wrapper, LAYER_FIELD_NAME);
      if (layerPts == null) continue;
      for (InstanceKey layerIK : layerPts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(layerIK);
        if (asin != null) ret.add(asin);
      }
    }
    return ret;
  }

  /**
   * Resolves the output's last-axis extents from the wrapped layer's {@code units} scaled by the
   * merge mode's width factor. Either part failing to resolve degrades to {@link UnresolvedDim},
   * keeping the rank.
   *
   * @param builder The propagation call graph builder.
   * @return The possible last-axis extents; never empty.
   */
  @Override
  protected Set<Dimension<?>> getOutputWidthDims(PropagationCallGraphBuilder builder) {
    Integer widthFactor = this.getWidthFactor(builder);

    Set<Dimension<?>> ret = HashSetFactory.make();
    if (widthFactor != null)
      for (AllocationSiteInNode carrier : this.getParameterCarriers(builder)) {
        Set<Long> unitValues =
            getPossibleLongValues(
                getInstanceFieldPointsToSet(builder, carrier, this.getUnitsFieldName()));
        if (unitValues == null) continue;
        for (Long units : unitValues)
          if (units != null) ret.add(new NumericDim(widthFactor * units.intValue()));
      }
    if (ret.isEmpty()) ret.add(UnresolvedDim.INSTANCE);
    return ret;
  }

  /**
   * Resolves the merge mode's width factor from the {@code merge_mode} field on the wrapper.
   *
   * @param builder The propagation call graph builder.
   * @return {@code 2} for {@code concat} (Keras's default when the argument is unsupplied), {@code
   *     1} for the element-wise modes, or {@code null} when the mode does not resolve to exactly
   *     one of them — {@code None} included, whose list-of-two-tensors result this generator does
   *     not model.
   */
  private Integer getWidthFactor(PropagationCallGraphBuilder builder) {
    Set<Object> modes = HashSetFactory.make();
    for (AllocationSiteInNode wrapper : super.getParameterCarriers(builder)) {
      Set<Object> values =
          getConstantValues(
              getInstanceFieldPointsToSet(builder, wrapper, MERGE_MODE_FIELD_NAME), true);
      if (values == null) return null;
      modes.addAll(values);
    }

    if (modes.isEmpty()) return 2;
    if (modes.size() != 1) return null;
    Object mode = modes.iterator().next();
    if (CONCAT_MERGE_MODE.equals(mode)) return 2;
    if (mode instanceof String && WIDTH_PRESERVING_MERGE_MODES.contains(mode)) return 1;
    return null;
  }
}
