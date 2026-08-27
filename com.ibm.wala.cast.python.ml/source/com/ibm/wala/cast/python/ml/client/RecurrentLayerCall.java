package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for the {@code __call__} on a recurrent Keras layer instance ({@code LSTM}, {@code
 * GRU}, and {@code SimpleRNN}). The layer's declared width becomes the output's last axis, read
 * back from the {@code units} argument the constructor summary stores on the instance, exactly as
 * {@code Dense} stores {@code units} and {@code Embedding} stores {@code output_dim}; {@code
 * return_sequences} decides whether the temporal axis survives (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>Given a rank-3 input {@code (B, T, F)}, the output is {@code (B, units)} when {@code
 * return_sequences} is false (Keras's default) and {@code (B, T, units)} when it is true. A {@code
 * return_sequences} the analysis cannot decide contributes both members, the union over the flag's
 * values, rather than a guess in either direction.
 *
 * <p>This is deliberately one generator rather than one per class: the recurrent classes differ in
 * what they compute and not in how they transform the type.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/LSTM">tf.keras.layers.LSTM</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RecurrentLayerCall extends DenseCall {

  private static final Logger LOGGER = Logger.getLogger(RecurrentLayerCall.class.getName());

  /** Rank of a recurrent layer's input: batch, time, and features. */
  private static final int RECURRENT_INPUT_RANK = 3;

  /** The constructor argument the model file stores on the layer instance. */
  private static final String RETURN_SEQUENCES_FIELD_NAME = "return_sequences";

  /**
   * Constructs a {@code RecurrentLayerCall} from a caller-side {@link PointsToSetVariable} (the
   * result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a recurrent layer instance.
   */
  public RecurrentLayerCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code RecurrentLayerCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public RecurrentLayerCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes from each rank-3 input shape, the layer's declared width, and the
   * {@code return_sequences} flag.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, one per rank-3 input shape, width, and flag value, or {@code
   *     null} if the input has no known shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = this.getInputShapes(builder);
    if (inputShapes == null) return null;
    // The engine's ⊥ (still-unresolved inputs) propagates as ⊥ so this evaluation ascends with
    // them instead of freezing a transient ⊤ into its join (wala/ML#758 clause 3); outside the
    // engine an empty set keeps the historical ⊤.
    if (inputShapes.isEmpty())
      return WorklistTypeResolver.active(builder) == null ? null : Collections.emptySet();

    Set<Dimension<?>> widths = this.getOutputWidthDims(builder);
    Set<Boolean> returnSequences = this.getPossibleReturnSequences(builder);

    Set<List<Dimension<?>>> outputShapes = new HashSet<>();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // Only a rank-3 input is a recurrent layer's input. Anything else is not this operation
      // being applied as documented, so say nothing rather than invent a shape.
      if (inputShape.size() != RECURRENT_INPUT_RANK) continue;

      for (Dimension<?> width : widths)
        for (boolean sequences : returnSequences) {
          List<Dimension<?>> outShape = new ArrayList<>();
          outShape.add(inputShape.get(0));
          if (sequences) outShape.add(inputShape.get(1));
          outShape.add(width);
          outputShapes.add(outShape);
        }
    }

    if (outputShapes.isEmpty()) {
      LOGGER.fine(
          () ->
              "No rank-"
                  + RECURRENT_INPUT_RANK
                  + " input shape for recurrent layer call at: "
                  + describe(this.getNode()));
      return null;
    }

    return outputShapes;
  }

  /**
   * Resolves the output's last-axis extents from the layer's declared width. An unresolvable width
   * is {@link UnresolvedDim}, following {@link Conv2DCall}: the rank is still known, and a
   * specification with the right rank and an unresolved axis beats no shape at all.
   *
   * @param builder The propagation call graph builder.
   * @return The possible last-axis extents; never empty.
   */
  protected Set<Dimension<?>> getOutputWidthDims(PropagationCallGraphBuilder builder) {
    Set<Long> unitValues = this.getPossibleUnits(builder);

    Set<Dimension<?>> ret = HashSetFactory.make();
    for (Long units : unitValues) if (units != null) ret.add(new NumericDim(units.intValue()));
    if (ret.isEmpty()) ret.add(UnresolvedDim.INSTANCE);
    return ret;
  }

  /**
   * The smallest positional index a {@code return_sequences} argument can take across the recurrent
   * constructors ({@code SimpleRNN}'s, where it is the 16th argument after {@code self}); a
   * constructor invocation with at least this many positional arguments may be supplying it
   * positionally.
   */
  private static final int MIN_RETURN_SEQUENCES_POSITION = 16;

  /**
   * Resolves the {@code return_sequences} flag from the field the constructor summary stores on the
   * instances carrying this call's parameters.
   *
   * @param builder The propagation call graph builder.
   * @return The flag's possible values; never empty. An argument no constructor call site supplies
   *     resolves to Keras's default of {@code false}; an undecidable one — supplied somewhere but
   *     with a value the analysis cannot see, such as a caller's default parameter value —
   *     contributes both values, the union rather than a guess in either direction.
   */
  protected Set<Boolean> getPossibleReturnSequences(PropagationCallGraphBuilder builder) {
    Collection<AllocationSiteInNode> carriers = this.getParameterCarriers(builder);

    Set<Boolean> ret = HashSetFactory.make();
    boolean undecidable = carriers.isEmpty();
    for (AllocationSiteInNode carrier : carriers) {
      Set<Boolean> values =
          getPossibleBooleanValues(
              getInstanceFieldPointsToSet(builder, carrier, RETURN_SEQUENCES_FIELD_NAME));
      LOGGER.fine(
          () ->
              "Possible `return_sequences` values: "
                  + values
                  + " for carrier: "
                  + describe(carrier)
                  + ".");
      if (values == null) undecidable = true;
      else if (values.isEmpty()) {
        // An empty field is ambiguous: the argument was never supplied (Keras's default holds), or
        // it was supplied with a value that never reached the points-to set. The constructor call
        // sites decide which: only an argument no site mentions is genuinely the default.
        boolean supplied = isReturnSequencesSupplied(builder, carrier.getNode());
        LOGGER.fine(
            () ->
                "Empty `return_sequences` field; syntactically supplied: "
                    + supplied
                    + " for carrier: "
                    + describe(carrier)
                    + ".");
        if (supplied) undecidable = true;
        else ret.add(false);
      } else ret.addAll(values);
    }
    if (undecidable) {
      ret.add(false);
      ret.add(true);
    }
    return ret;
  }

  /**
   * Whether any call site of the given constructor node supplies a {@code return_sequences}
   * argument, by keyword or positionally.
   *
   * @param builder The propagation call graph builder.
   * @param constructorNode The constructor's synthetic {@code do} node.
   * @return {@code true} iff some invocation dispatching to it mentions the argument.
   */
  private static boolean isReturnSequencesSupplied(
      PropagationCallGraphBuilder builder, CGNode constructorNode) {
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, constructorNode)) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction call = (PythonInvokeInstruction) callerInvoke.snd;
      if (call.getUse(RETURN_SEQUENCES_FIELD_NAME) != -1) return true;
      if (call.getNumberOfPositionalParameters() - 1 >= MIN_RETURN_SEQUENCES_POSITION) return true;
    }
    return false;
  }

  /**
   * The instances carrying this call's constructor arguments: the receiver itself for a plain
   * recurrent layer. {@link BidirectionalCall} redirects this to the wrapped layer, which is where
   * a {@code Bidirectional} keeps them.
   *
   * @param builder The propagation call graph builder.
   * @return The allocation sites of the parameter-carrying instances; possibly empty.
   */
  protected Collection<AllocationSiteInNode> getParameterCarriers(
      PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());

    List<AllocationSiteInNode> ret = new ArrayList<>();
    if (selfPts != null)
      for (InstanceKey selfIK : selfPts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(selfIK);
        if (asin != null) ret.add(asin);
      }
    return ret;
  }
}
