package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A representation of the `tf.convert_to_tensor()` API in TensorFlow.
 *
 * <p>This function converts Python objects of various types to Tensor objects. It accepts Tensor
 * objects, numpy arrays, Python lists, and Python scalars. The value-argument shape/dtype inference
 * lives in {@link ValueExtractingTensorGenerator}; this op adds a {@code dtype_hint} fallback and
 * has no explicit shape argument.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor">tf.convert_to_tensor
 *     API</a>.
 */
public class ConvertToTensor extends ValueExtractingTensorGenerator {

  protected enum Parameters {
    VALUE,
    DTYPE,
    /**
     * Optional element type for the returned tensor, used when <code>dtype</code> is <code>None
     * </code>.
     *
     * <p>Need to consider this when inferring default dtypes.
     *
     * @see <a href="https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor#dtype_hint">
     *     <code>dtype_hint</code> parameter</a>.
     */
    DTYPE_HINT,
    NAME,
    AS_REF,
    PREFERRED_DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ConvertToTensor(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ConvertToTensor(CGNode node) {
    super(node);
  }

  /**
   * A conversion returns its {@code value} unchanged when that value is already a tensor, so the
   * result's type is the value's, and this declares a {@link TypeFeedKind#PASS_THROUGH} feed over
   * each caller's {@code value} argument (<a
   * href="https://github.com/wala/ML/issues/947">wala/ML#947</a>). Without one, a value typed only
   * by dataflow, such as a Keras layer's call result, has an empty points-to set, so this
   * generator's seed is the pure ⊤ while the pass-through edge still delivers the value's real
   * type, and the join keeps both. Every summary that returns {@code convert_to_tensor(x)} ({@code
   * tf.tanh} among them) inherits that member. A declared feed replaces the ⊤ seed with the
   * operand's composed type instead (<a
   * href="https://github.com/wala/ML/issues/736">wala/ML#736</a>).
   *
   * <p>The feed is withheld entirely when any caller passes anything beyond the value, since a
   * {@code dtype}, a {@code dtype_hint} or any keyword can make the result's dtype differ from the
   * value's, and a feed describes every calling context at once. It is also withheld unless this
   * generator is anchored in the conversion's own summary node, the only frame whose callers pass
   * {@code value} as their first argument.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The pass-through feed over the callers' {@code value} arguments, or {@code null} when
   *     it is withheld or no caller is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    CGNode node = this.getNode();
    if (!node.getMethod()
        .getDeclaringClass()
        .getReference()
        .equals(CONVERT_TO_TENSOR.getDeclaringClass())) return null;

    List<PointerKey> operands = new ArrayList<>();
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, node)) {
      // The callee (or, from a summary body, the receiver) and the value: nothing else.
      if (callerInvoke.snd.getNumberOfUses() != 2) return null;
      operands.add(
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(callerInvoke.fst, callerInvoke.snd.getUse(1)));
    }
    return operands.isEmpty() ? null : new TypeFeed(TypeFeedKind.PASS_THROUGH, operands);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If the dtype argument is not specified, then the type is inferred from the type of value,
    // unless dtype_hint is provided.
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.DTYPE_HINT.getIndex(), Parameters.DTYPE_HINT.getName(), true);
    Set<DType> defaultDTypes = super.getDefaultDTypes(builder);

    if (valNum <= 0) return defaultDTypes;

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, Parameters.DTYPE_HINT.getIndex(), Parameters.DTYPE_HINT.getName());

    if (pointsToSet == null || pointsToSet.isEmpty()) {
      // If the argument dtype hint is not specified.
      return defaultDTypes;
    } else {
      // The dtype points-to set is non-empty, meaning that the dtype hint was explicitly set. If
      // the
      // conversion to dtype_hint is not possible, this argument has no effect. Get the dtypes from
      // the points-to set.
      Set<DType> dTypesFromDTypeHintArgument =
          this.getDTypesFromDTypeArgument(builder, pointsToSet);

      // For each possible dtype from dtype hint, check if it is compatible with default dtypes.
      Set<DType> compatibleDTypes = EnumSet.noneOf(DType.class);

      for (DType dTypeFromDTypeHint : dTypesFromDTypeHintArgument)
        for (DType defaultDType : defaultDTypes)
          if (defaultDType.canConvertTo(dTypeFromDTypeHint))
            compatibleDTypes.add(dTypeFromDTypeHint);

      // No compatible dtypes found, return the default dtypes.
      if (!compatibleDTypes.isEmpty()) return compatibleDTypes;
      else return defaultDTypes;
    }
  }
}
