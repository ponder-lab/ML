package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for a call on a {@code tf.keras.losses.SparseCategoricalCrossentropy} instance (<a
 * href="https://github.com/wala/ML/issues/951">wala/ML#951</a>): the loss has {@code y_pred}'s
 * dtype, and its shape is {@code y_pred}'s without the trailing class axis under {@code
 * Reduction.NONE}, or a scalar under the summing reductions and the default {@code auto}, which
 * resolves to {@code sum_over_batch_size} outside a distribution strategy (TensorFlow 2.9.3). The
 * {@code reduction} is read from the instance the constructor summary stored it on, as {@link
 * DenseCall} reads {@code units}; an instance whose {@code reduction} does not resolve to a string
 * constant is read as the default. A call whose {@code y_pred} the substrate cannot type declares a
 * {@link TypeFeedKind#TRANSFORM} feed carrying the same rule, so a prediction typed only by the
 * dataflow (a model's output) still types its loss.
 */
public class SparseCategoricalCrossentropyCall extends TensorGenerator {

  private static final Logger LOGGER =
      Logger.getLogger(SparseCategoricalCrossentropyCall.class.getName());

  /** The instance field the constructor summary stores the {@code reduction} argument in. */
  private static final String REDUCTION_FIELD_NAME = "reduction";

  /** The {@code Reduction.NONE} constant's value, as the module summary defines it. */
  private static final String REDUCTION_NONE = "none";

  /**
   * Parameter positions of the {@code __call__} summary after the callable slot: {@code self},
   * {@code y_true}, {@code y_pred} (and {@code sample_weight}, not consumed).
   */
  protected enum Parameters {
    SELF,
    Y_TRUE,
    Y_PRED;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseCategoricalCrossentropyCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code __call__} method.
   */
  public SparseCategoricalCrossentropyCall(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> predictions = this.predictionShapes(builder);
    if (predictions == null || predictions.isEmpty()) return predictions;
    Reduction reduction = this.reduction(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : predictions) {
      Set<List<Dimension<?>>> losses = lossShapes(shape, reduction);
      if (losses == null) return null;
      ret.addAll(losses);
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, Parameters.Y_PRED.getIndex(), Parameters.Y_PRED.getName());
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty() && !dtypes.equals(EnumSet.of(DType.UNKNOWN)))
        return dtypes;
    }
    Set<DType> viaCallers =
        this.getArgumentDTypesViaCallers(
            builder, Parameters.Y_PRED.getIndex(), Parameters.Y_PRED.getName());
    return viaCallers == null || viaCallers.isEmpty() ? EnumSet.of(DType.UNKNOWN) : viaCallers;
  }

  /**
   * The {@code y_pred} argument's shapes: its points-to union first, then the per-context caller
   * walk, as {@link Reshape} reads its input.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The prediction shapes, {@code null} when unknown.
   */
  private Set<List<Dimension<?>>> predictionShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, Parameters.Y_PRED.getIndex(), Parameters.Y_PRED.getName());
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> fromValue = this.getShapesOfValue(builder, pts);
      if (fromValue != null && !fromValue.isEmpty()) return fromValue;
    }
    ShapeResult viaCallers =
        this.getArgumentShapeResultViaCallers(
            builder, Parameters.Y_PRED.getIndex(), Parameters.Y_PRED.getName());
    if (!viaCallers.members().isEmpty()) return viaCallers.members();
    return null;
  }

  /**
   * The loss shapes for one prediction shape: the prediction without its trailing class axis under
   * {@code Reduction.NONE}, a scalar otherwise.
   *
   * @param prediction The prediction's dimensions.
   * @param reduction The instance's reduction.
   * @return The loss shapes, or {@code null} for a rank-0 prediction, which has no class axis, and
   *     for a reduction that was passed but did not resolve, where the shape is one of two.
   */
  private static Set<List<Dimension<?>>> lossShapes(
      List<Dimension<?>> prediction, Reduction reduction) {
    if (prediction.isEmpty() || reduction == Reduction.UNRESOLVED) return null;
    if (reduction == Reduction.SUMMING) return Collections.singleton(Collections.emptyList());
    return Collections.singleton(new ArrayList<>(prediction.subList(0, prediction.size() - 1)));
  }

  /** How the instance reduces, as far as the substrate says. */
  private enum Reduction {
    /** {@code Reduction.NONE}: the loss keeps the predictions' shape without the class axis. */
    NONE,
    /** A summing reduction, or the default, which sums: a scalar. */
    SUMMING,
    /** A reduction was passed but resolves to no string constant: the shape is one of two. */
    UNRESOLVED
  }

  /**
   * The instance's reduction, read from the field the constructor summary stored the argument in.
   * Three-valued, because the default and an unresolved argument must not read alike: an instance
   * whose stored {@code reduction} resolves to {@code "none"} reduces with {@code Reduction.NONE};
   * one whose stored value resolves to another string, or whose constructor was called without a
   * {@code reduction}, sums; and one whose constructor was passed a {@code reduction} that resolves
   * to no string constant (a variable, a configuration value) is unresolved, since asserting the
   * default there would be a wrong answer whenever the value is {@code "none"}. Several instances
   * that disagree are unresolved too, and so is a call whose receiver resolves to no instance at
   * all: the default is justified only by an instance that was constructed without a reduction.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The reduction.
   */
  private Reduction reduction(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());
    if (selfPts == null) return Reduction.UNRESOLVED;
    Reduction ret = null;
    for (InstanceKey selfIk : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIk);
      if (selfAsin == null) continue;
      Reduction ofInstance = this.reductionOf(builder, selfAsin);
      if (ret == null) ret = ofInstance;
      else if (ret != ofInstance) return Reduction.UNRESOLVED;
    }
    return ret == null ? Reduction.UNRESOLVED : ret;
  }

  /**
   * One instance's reduction: the string constants its stored {@code reduction} points to, or, when
   * it points to none, whether the constructor call that allocated it passed a {@code reduction} at
   * all (positionally, as the third argument after the callable and {@code from_logits}, or by
   * keyword).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param selfAsin The instance's allocation, in the constructor summary's node.
   * @return The instance's reduction.
   */
  private Reduction reductionOf(
      PropagationCallGraphBuilder builder, AllocationSiteInNode selfAsin) {
    FieldReference reductionRef =
        FieldReference.findOrCreate(
            selfAsin.concreteType().getReference(),
            findOrCreateAsciiAtom(REDUCTION_FIELD_NAME),
            Root);
    IField field = builder.getClassHierarchy().resolveField(reductionRef);
    if (field != null) {
      PointerKey fieldKey = builder.getPointerKeyForInstanceField(selfAsin, field);
      OrdinalSet<InstanceKey> reductionPts = builder.getPointerAnalysis().getPointsToSet(fieldKey);
      if (reductionPts != null && !reductionPts.isEmpty()) {
        Set<Object> values = getConstantValues(reductionPts, true);
        if (values == null || values.isEmpty()) return Reduction.UNRESOLVED;
        Reduction ret = null;
        for (Object value : values) {
          Reduction ofValue = REDUCTION_NONE.equals(value) ? Reduction.NONE : Reduction.SUMMING;
          if (ret == null) ret = ofValue;
          else if (ret != ofValue) return Reduction.UNRESOLVED;
        }
        LOGGER.fine(() -> "Reduction resolves to " + values + ".");
        return ret;
      }
    }
    // Nothing stored: the default, unless the constructor call passed a reduction the analysis
    // could not read.
    for (Pair<CGNode, SSAAbstractInvokeInstruction> ctorCall :
        getCallerInvokes(builder, selfAsin.getNode())) {
      if (!(ctorCall.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction call = (PythonInvokeInstruction) ctorCall.snd;
      if (call.getNumberOfPositionalParameters() > 2 || call.getUse(REDUCTION_FIELD_NAME) != -1) {
        LOGGER.fine(() -> "A reduction is passed but resolves to no constant; the shape is ⊤.");
        return Reduction.UNRESOLVED;
      }
    }
    return Reduction.SUMMING;
  }

  /**
   * Declares a {@link TypeFeedKind#TRANSFORM} feed over {@code y_pred} carrying the loss-shape
   * rule, so a prediction typed only by the dataflow types its loss the way a substrate-typed one
   * does.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The rule-carrying feed over {@code y_pred}'s key, or {@code null} when the argument
   *     cannot be located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    CGNode node = this.getNode();
    if (node == null || node.getIR() == null) return null;
    int predVn =
        this.getArgumentValueNumber(
            builder, Parameters.Y_PRED.getIndex(), Parameters.Y_PRED.getName(), true);
    if (predVn <= 0 || predVn == Integer.MAX_VALUE) return null;
    PointerKey prediction =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, predVn);
    Reduction reduction = this.reduction(builder);
    return new TypeFeed(
        TypeFeedKind.TRANSFORM, List.of(prediction), input -> lossShapes(input, reduction));
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
