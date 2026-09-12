package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
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
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for a call on a {@code tf.keras.applications} model (wala/ML#896): a pretrained
 * architecture such as {@code MobileNetV2} or {@code ResNet50}, used as a backbone.
 *
 * <p>The output rank is fixed by the constructor alone, whatever the input: with {@code
 * include_top=True} (the default) the model classifies, so the output is rank 2; with {@code
 * include_top=False} and a {@code pooling} of {@code "avg"} or {@code "max"} the feature map is
 * pooled to rank 2; with {@code include_top=False} and no pooling the output is the rank-4 feature
 * map. The batch axis is the input's; every other axis is {@link UnresolvedDim}, since the spatial
 * and channel extents are fixed integers of the architecture that this generator does not compute
 * (wala/ML#721). Recovering the rank is what a downstream {@code Flatten} then {@code Dense} needs
 * to reach the final shape; without it the call result carried no shape at all and every value
 * downstream lost its rank, the same class of loss {@link Conv2DCall} documents.
 *
 * <p>A constructor argument the analysis cannot decide (a non-constant {@code include_top}, a
 * non-constant or unrecognized {@code pooling}, or instances that disagree) yields unknown rank
 * rather than a guess.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/applications">tf.keras.applications</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class KerasApplicationCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(KerasApplicationCall.class.getName());

  /** The stored {@code include_top} constructor argument; unbound means Keras's default, true. */
  private static final String INCLUDE_TOP_FIELD_NAME = "include_top";

  /** The stored {@code pooling} constructor argument; unbound means Keras's default, none. */
  private static final String POOLING_FIELD_NAME = "pooling";

  /** The pooling modes that reduce the feature map to a vector. */
  private static final Set<String> POOLING_MODES = Set.of("avg", "max");

  /** Rank of a feature-map output: batch, two spatial axes, and channels. */
  private static final int FEATURE_MAP_RANK = 4;

  /** Rank of a pooled or classifying output: batch and features. */
  private static final int VECTOR_RANK = 2;

  public KerasApplicationCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public KerasApplicationCall(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Integer rank = this.outputRank(builder);
    if (rank == null) return null;

    // The batch axis is the input's leading axis when the input's shape is known; otherwise it is
    // a fixed extent this generator cannot see. The rank does not depend on the input.
    Set<Dimension<?>> batches = HashSetFactory.make();
    Set<List<Dimension<?>>> inputShapes = this.getArgumentShapesWithFallback(builder, 1, "inputs");
    if (inputShapes != null)
      for (List<Dimension<?>> inputShape : inputShapes)
        batches.add(inputShape.isEmpty() ? UnresolvedDim.INSTANCE : inputShape.get(0));
    if (batches.isEmpty()) batches.add(UnresolvedDim.INSTANCE);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Dimension<?> batch : batches) {
      List<Dimension<?>> shape = new ArrayList<>();
      shape.add(batch);
      shape.addAll(Collections.nCopies(rank - 1, UnresolvedDim.INSTANCE));
      ret.add(shape);
    }
    LOGGER.fine(() -> "Keras application call resolved to rank " + rank + ": " + ret + ".");
    return ret;
  }

  /**
   * Decides the output rank from the {@code include_top} and {@code pooling} the constructor stored
   * on every receiver instance.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for points-to lookup.
   * @return The rank, or {@code null} when an argument is not statically decided or the receiver
   *     instances disagree.
   */
  private Integer outputRank(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return null;

    Integer rank = null;
    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return null;

      Boolean includeTop = this.storedIncludeTop(builder, selfAsin);
      if (includeTop == null) return null;

      Boolean pooled = this.storedPooling(builder, selfAsin);
      if (pooled == null) return null;

      int instanceRank = includeTop || pooled ? VECTOR_RANK : FEATURE_MAP_RANK;
      if (rank != null && rank != instanceRank) return null;
      rank = instanceRank;
    }
    return rank;
  }

  /**
   * Reads the stored {@code include_top}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for points-to lookup.
   * @param instance The model instance the constructor allocated.
   * @return The single boolean, Keras's default {@code true} when the constructor call omits the
   *     argument, or {@code null} when the argument is supplied with a value the analysis cannot
   *     read or is not a single boolean.
   */
  private Boolean storedIncludeTop(
      PropagationCallGraphBuilder builder, AllocationSiteInNode instance) {
    OrdinalSet<InstanceKey> pts =
        getInstanceFieldPointsToSet(builder, instance, INCLUDE_TOP_FIELD_NAME);
    if (pts == null || pts.isEmpty()) {
      return Boolean.FALSE.equals(this.supplied(builder, instance, INCLUDE_TOP_FIELD_NAME))
          ? Boolean.TRUE
          : null;
    }
    Set<Boolean> values = getPossibleBooleanValues(pts);
    return values != null && values.size() == 1 ? values.iterator().next() : null;
  }

  /**
   * Reads whether the stored {@code pooling} reduces the feature map.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for points-to lookup.
   * @param instance The model instance the constructor allocated.
   * @return {@code true} for {@code "avg"} or {@code "max"}, {@code false} for an omitted or {@code
   *     None} pooling, and {@code null} when the argument is supplied with a value the analysis
   *     cannot read.
   */
  private Boolean storedPooling(
      PropagationCallGraphBuilder builder, AllocationSiteInNode instance) {
    OrdinalSet<InstanceKey> pts =
        getInstanceFieldPointsToSet(builder, instance, POOLING_FIELD_NAME);
    if (pts == null || pts.isEmpty()) {
      return Boolean.FALSE.equals(this.supplied(builder, instance, POOLING_FIELD_NAME))
          ? Boolean.FALSE
          : null;
    }
    Set<Object> values = getConstantValues(pts, true);
    if (values == null || values.size() != 1) return null;
    Object value = values.iterator().next();
    // Keras accepts exactly `None`, "avg" and "max"; a running program supplies nothing else.
    return value == null ? Boolean.FALSE : POOLING_MODES.contains(value) ? Boolean.TRUE : null;
  }

  /**
   * Whether the constructor call that allocated the instance supplies the named argument, read from
   * the call site rather than the stored field: an omitted argument and a supplied one whose value
   * the analysis cannot represent (a runtime flag) leave the same empty field behind, and only the
   * first may take the API default (the wala/ML#865 distinction).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param instance The model instance, allocated in the constructor's synthetic body.
   * @param name The constructor parameter's name.
   * @return {@code TRUE}, {@code FALSE}, or {@code null} when the call sites cannot tell.
   */
  private Boolean supplied(
      PropagationCallGraphBuilder builder, AllocationSiteInNode instance, String name) {
    CGNode constructor = instance.getNode();
    return isArgumentSyntacticallySuppliedAt(
        builder, constructor, positionalIndexOf(constructor, name), name);
  }

  /**
   * The 0-based positional index (the receiver excluded) of the named parameter of a summarized
   * constructor, from the local names its summary declares.
   *
   * @param constructor The constructor's {@link CGNode}.
   * @param name The parameter name.
   * @return The index, or {@code -1} when no parameter carries the name.
   */
  private static int positionalIndexOf(CGNode constructor, String name) {
    if (constructor.getIR() == null) return -1;
    int parameters = constructor.getIR().getNumberOfParameters();
    for (int i = 0; i < parameters; i++) {
      String[] names = constructor.getIR().getLocalNames(0, constructor.getIR().getParameter(i));
      if (names != null) for (String candidate : names) if (name.equals(candidate)) return i - 1;
    }
    return -1;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Keras models compute in the floating-point default whatever the input dtype, as `DenseCall`
    // assumes for layers.
    return EnumSet.of(DType.FLOAT32);
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
