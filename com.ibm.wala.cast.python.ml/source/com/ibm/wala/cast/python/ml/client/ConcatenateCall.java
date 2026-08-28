package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

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
import java.util.Set;

/**
 * Generator for the {@code __call__} on a {@code tf.keras.layers.Concatenate} instance. The
 * computation is {@link Concat}'s over the list the call receives; what differs is the geometry:
 * the tensors list arrives as the {@code inputs} argument after {@code self}, and the axis is the
 * constructor argument the summary stores on the layer instance rather than a call argument, with
 * Keras's default of {@code -1} (<a href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * <p>An empty {@code axis} field is disambiguated at the constructor call sites, mirroring {@link
 * RecurrentLayerCall}'s {@code return_sequences} handling: an argument no site mentions is
 * genuinely the default, while one supplied somewhere with a value the analysis cannot see leaves
 * the axis unresolved, and the shape degrades to ⊤ rather than concatenating along a guessed axis.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Concatenate">tf.keras.layers.Concatenate</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ConcatenateCall extends KerasConcatenate {

  /** The constructor argument the model file stores on the layer instance. */
  private static final String AXIS_FIELD_NAME = "axis";

  /** The {@code inputs} argument's positional index, excluding the trampoline's {@code func}. */
  private static final int INPUTS_PARAMETER_INDEX = 1;

  /**
   * Constructs a {@code ConcatenateCall} from a caller-side {@link PointsToSetVariable} (the result
   * of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code Concatenate} instance.
   */
  public ConcatenateCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code ConcatenateCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public ConcatenateCall(CGNode node) {
    super(node);
  }

  /**
   * @return {@value #INPUTS_PARAMETER_INDEX}; the call receives the tensors list after {@code
   *     self}.
   */
  @Override
  protected int getValuesParameterIndex() {
    return INPUTS_PARAMETER_INDEX;
  }

  /**
   * Resolves the axis from the field the constructor summary stores on the layer instance rather
   * than from a call argument.
   *
   * @param builder The propagation call graph builder.
   * @return The resolved axis, or {@code null} when it does not resolve to exactly one integer
   *     across the receiver instances.
   */
  @Override
  protected Integer resolveConstantAxis(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return null;

    Set<Integer> axes = HashSetFactory.make();
    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) return null;

      Set<Long> values =
          getPossibleLongValues(getInstanceFieldPointsToSet(builder, selfAsin, AXIS_FIELD_NAME));
      if (values == null) return null; // supplied but not statically resolvable → ⊤.
      if (values.isEmpty()) {
        // An empty field is ambiguous: the argument was never supplied (Keras's default holds), or
        // it was supplied with a value that never reached the points-to set. The constructor call
        // sites decide which: only an argument no site mentions is genuinely the default.
        if (isAxisSupplied(builder, selfAsin.getNode())) return null;
        axes.add(this.getDefaultAxis());
      } else
        for (Long value : values) {
          if (value == null) return null;
          axes.add(value.intValue());
        }
    }

    return axes.size() == 1 ? axes.iterator().next() : null;
  }

  /**
   * Whether any call site of the given constructor node supplies an {@code axis} argument, by
   * keyword or positionally ({@code axis} is the constructor's first argument after {@code self}).
   *
   * @param builder The propagation call graph builder.
   * @param constructorNode The constructor's synthetic {@code do} node.
   * @return {@code true} iff some invocation dispatching to it mentions the argument.
   */
  private static boolean isAxisSupplied(
      PropagationCallGraphBuilder builder, CGNode constructorNode) {
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, constructorNode)) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction call = (PythonInvokeInstruction) callerInvoke.snd;
      if (call.getUse(AXIS_FIELD_NAME) != -1) return true;
      if (call.getNumberOfPositionalParameters() - 1 >= 1) return true;
    }
    return false;
  }
}
