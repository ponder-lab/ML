package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_TYPE;
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
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.random.categorical(logits, num_samples, dtype=None, seed=None,
 * name=None)}, which draws {@code num_samples} class indices for each row of a {@code (batch,
 * num_classes)} logits tensor. The result is a fresh {@code (batch, num_samples)} tensor of {@code
 * dtype}, {@code int64} by default.
 *
 * <p>The summary used to return {@code logits} itself, so a draw carried the logits' shape and
 * float dtype. A sampling loop feeds the draw back as the next step's input, where an embedding
 * lookup adds a dimension, so the aliased draw grew one rank per round.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/random/categorical">
 *     tf.random.categorical</a>
 */
public class Categorical extends TensorGenerator {

  /**
   * Parameter positions and keyword names, after the implicit {@code self} receiver, matching
   * {@code tensorflow.xml}'s {@code paramNames}.
   */
  protected enum Parameters {
    /** The 2-D {@code (batch, num_classes)} unnormalized log-probabilities. */
    LOGITS,

    /** The number of independent draws per row. */
    NUM_SAMPLES,

    /** The integer type of the draws; {@code int64} when omitted. */
    DTYPE,

    /** The random seed; not consumed by this generator. */
    SEED,

    /** The operation's name; not consumed by this generator. */
    NAME;

    /**
     * The keyword name of this parameter.
     *
     * @return The lowercased enum name.
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    /**
     * The positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public Categorical(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Categorical(CGNode node) {
    super(node);
  }

  /**
   * The draw's shape: the logits' batch extent followed by {@code num_samples}. A logits shape of
   * any rank other than 2 raises at run time and contributes nothing.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The {@code (batch, num_samples)} shapes, or {@code null} when the logits' shape is
   *     unknown.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> logitsShapes = null;
    OrdinalSet<InstanceKey> logitsPts =
        this.getArgumentPointsToSet(
            builder, Parameters.LOGITS.getIndex(), Parameters.LOGITS.getName());
    if (logitsPts != null && !logitsPts.isEmpty())
      logitsShapes = this.getShapesOfValue(builder, logitsPts);
    if (logitsShapes == null || logitsShapes.isEmpty())
      // An overlay-resolved logits value (a slice, an elementwise result) can have no points-to
      // allocation at the synthetic node; read it in the caller's frame (wala/ML#718).
      logitsShapes =
          this.getArgumentShapeResultViaCallers(
                  builder, Parameters.LOGITS.getIndex(), Parameters.LOGITS.getName())
              .toLegacy();
    if (logitsShapes == null || logitsShapes.isEmpty()) return null;

    Dimension<?> samples = this.numSamplesAxis(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> logits : logitsShapes) {
      if (logits == null) return null;
      if (logits.size() == 2) ret.add(List.of(logits.get(0), samples));
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * The {@code num_samples} extent: its constant value, or, per wala/ML#721, {@link DynamicDim}
   * when it is a tensor (TensorFlow's static shape reports {@code None} there) and {@link
   * UnresolvedDim} otherwise.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The extent of the draw's second axis.
   */
  private Dimension<?> numSamplesAxis(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, Parameters.NUM_SAMPLES.getIndex(), Parameters.NUM_SAMPLES.getName());
    if (pts == null || pts.isEmpty()) return UnresolvedDim.INSTANCE;
    Set<Integer> values = HashSetFactory.make();
    boolean tensor = false;
    for (InstanceKey key : pts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(key);
      if (asin != null && asin.concreteType().getReference().equals(TENSOR_TYPE)) tensor = true;
    }
    for (Object value : getConstantValues(pts, false))
      if (value instanceof Number) values.add(((Number) value).intValue());
    if (values.size() == 1 && !tensor) return new NumericDim(values.iterator().next());
    return tensor ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE;
  }

  /**
   * The draws' type when no {@code dtype} is passed: TensorFlow's default, {@code int64}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code int64}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.INT64);
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
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
