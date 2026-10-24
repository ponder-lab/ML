package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.debug.UnimplementedError;
import java.util.Set;

/**
 * The client-facing authority on the rank of a {@code tf.reshape} call's shape operand (<a
 * href="https://github.com/wala/ML/issues/1147">wala/ML#1147</a>). Eagerly, {@code Reshape} accepts
 * a rank-0 shape such as {@code -1}; while a graph is being built, its shape inference requires a
 * rank-1 shape. A client deciding whether a function may run as a graph asks this rather than
 * reading the operand's points-to set and tensor types itself.
 *
 * <p>The answer is one of three {@link Rank}s, and {@link Rank#UNKNOWN} is never reported as {@link
 * Rank#RANK_1}. In particular, a computed Python int has no instance key (<a
 * href="https://github.com/wala/ML/issues/805">wala/ML#805</a>), so a shape operand that is one has
 * an empty points-to set, and it reads as unknown rather than as a vector.
 *
 * @see TensorFlowApis
 */
public final class ShapeOperandRank {

  /** The rank of a shape operand. */
  public enum Rank {
    /** Every value the operand may hold is known to be rank 1. */
    RANK_1,

    /** Some value the operand may hold is known to be rank 0. The others may be anything. */
    MAY_BE_RANK_0,

    /** Neither holds. */
    UNKNOWN
  }

  /** The keyword of {@code tf.reshape}'s shape operand. */
  private static final String SHAPE_KEYWORD = "shape";

  /**
   * The positional slot of {@code tf.reshape}'s shape operand: slot 0 is the callee, slot 1 the
   * tensor.
   */
  private static final int SHAPE_SLOT = 2;

  private ShapeOperandRank() {}

  /** What the values a key may hold show about its rank. */
  private static final class Evidence {

    /** Some value is known to be rank 0. */
    private boolean zero;

    /** Some value is known to be rank 1. */
    private boolean one;

    /** Some value is not known to be rank 1. */
    private boolean other;
  }

  /**
   * Returns the rank of the shape operand of a call to {@code tf.reshape}, by position or by its
   * {@code shape} keyword. The caller identifies the call as one to {@code tf.reshape}, as {@link
   * TensorFlowApis} names it. A call whose operand cannot be aligned, because a starred argument
   * comes at or before it, or that passes none, is {@link Rank#UNKNOWN}.
   *
   * @param builder The builder whose pointer analysis holds the operand's values.
   * @param analysis The solved {@link TensorTypeAnalysis}, which gives a tensor operand's rank.
   * @param node The node whose IR holds the call.
   * @param call The call.
   * @return The operand's rank.
   */
  public static Rank ofReshape(
      PropagationCallGraphBuilder builder,
      TensorTypeAnalysis analysis,
      CGNode node,
      PythonInvokeInstruction call) {
    int starred = call.firstStarredPosition();
    int operand;
    if (starred != -1 && starred <= SHAPE_SLOT) return Rank.UNKNOWN;
    if (call.getNumberOfPositionalParameters() > SHAPE_SLOT) operand = call.getUse(SHAPE_SLOT);
    else if (call.getKeywords().contains(SHAPE_KEYWORD)) operand = call.getUse(SHAPE_KEYWORD);
    else return Rank.UNKNOWN;
    return of(
        builder,
        analysis,
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, operand));
  }

  /**
   * Returns the rank of a value used as a shape.
   *
   * @param builder The builder whose pointer analysis holds the value.
   * @param analysis The solved {@link TensorTypeAnalysis}.
   * @param value The value's pointer key.
   * @return Its rank.
   */
  private static Rank of(
      PropagationCallGraphBuilder builder, TensorTypeAnalysis analysis, PointerKey value) {
    Evidence evidence = new Evidence();
    boolean needsTensorEvidence = false;
    for (InstanceKey member : builder.getPointerAnalysis().getPointsToSet(value)) {
      if (member instanceof ConstantKey<?> constant) {
        if (isScalarConstant(constant)) evidence.zero = true;
        else evidence.other = true;
      } else if (PythonTypes.isSequence(allocatedType(member))) {
        // A list or tuple is rank 1 when its elements are scalars; an element that is itself a
        // sequence or a tensor of nonzero rank stacks into a higher rank.
        if (elementsAreScalars(builder, analysis, member)) evidence.one = true;
        else evidence.other = true;
      } else needsTensorEvidence = true;
    }
    // A member that is neither a constant nor a sequence is classified only by the tensor-type
    // analysis. With no tensor state, it may be anything. An empty points-to set, as a computed int
    // has, gives no evidence of rank 1, so it reads as unknown below.
    if (!addTensorEvidence(builder, analysis, value, evidence) && needsTensorEvidence)
      evidence.other = true;
    if (evidence.zero) return Rank.MAY_BE_RANK_0;
    if (evidence.one && !evidence.other) return Rank.RANK_1;
    return Rank.UNKNOWN;
  }

  /**
   * Whether a constant is a Python number. A {@code bool} is an {@code int} in Python, so it is
   * one.
   *
   * @param constant The constant.
   * @return {@code true} iff it is a number or a {@code bool}.
   */
  private static boolean isScalarConstant(ConstantKey<?> constant) {
    return constant.getValue() instanceof Number || constant.getValue() instanceof Boolean;
  }

  /**
   * Whether every element of a list or tuple may be only a scalar. An element the analysis gives no
   * evidence for, with no points-to members and no tensor state, is read as one, as is an element
   * the container's keys may not name. Either may in fact be a non-scalar, but a shape with a
   * non-scalar element has rank 2 or more, or no rank at all, and {@code Reshape} rejects it
   * eagerly as well as in a graph. So reading such an element as a scalar can misreport only a call
   * that fails in both modes. An element with no evidence is today typically a computed primitive,
   * which has no instance key (<a href="https://github.com/wala/ML/issues/805">wala/ML#805</a>),
   * such as {@code x.shape[0]}.
   *
   * @param builder The builder.
   * @param analysis The solved {@link TensorTypeAnalysis}.
   * @param sequence The list or tuple.
   * @return {@code true} iff no element may be a sequence or a tensor of nonzero or unknown rank.
   */
  private static boolean elementsAreScalars(
      PropagationCallGraphBuilder builder, TensorTypeAnalysis analysis, InstanceKey sequence) {
    for (PointerKey element : ContainerElements.of(builder, sequence).allElements()) {
      for (InstanceKey member : builder.getPointerAnalysis().getPointsToSet(element))
        if (member instanceof ConstantKey<?> constant) {
          if (!isScalarConstant(constant)) return false;
        } else if (PythonTypes.isContainer(allocatedType(member))) return false;
      Evidence evidence = new Evidence();
      addTensorEvidence(builder, analysis, element, evidence);
      if (evidence.one || evidence.other) return false;
    }
    return true;
  }

  /**
   * Adds what the tensor-type analysis gives a key to its evidence: a type of rank 0 or rank 1 is
   * evidence of that rank, and a type of another or unknown rank, or an unknown tensor (⊤), is
   * evidence of neither.
   *
   * @param builder The builder.
   * @param analysis The solved {@link TensorTypeAnalysis}.
   * @param key The key.
   * @param evidence The evidence to add to.
   * @return {@code true} iff the analysis gives the key any tensor state.
   */
  private static boolean addTensorEvidence(
      PropagationCallGraphBuilder builder,
      TensorTypeAnalysis analysis,
      PointerKey key,
      Evidence evidence) {
    PointsToSetVariable variable = variable(builder, key);
    // A key the analysis's flow graph does not hold has no state, rather than an empty one.
    if (variable == null || !analysis.getProblem().getFlowGraph().containsNode(variable))
      return false;
    Set<TensorType> state = analysis.getOutState(variable);
    if (state == null) {
      // An unknown tensor (⊤): its rank is unknown.
      evidence.other = true;
      return true;
    }
    boolean any = false;
    for (TensorType type : state) {
      if (type.isNone()) continue;
      any = true;
      if (type.getDims() == null || type.getDims().size() > 1) evidence.other = true;
      else if (type.getDims().isEmpty()) evidence.zero = true;
      else evidence.one = true;
    }
    return any;
  }

  /**
   * The type an instance key allocates, for the two key kinds that name one.
   *
   * @param key The instance key.
   * @return The allocated type, or {@code null} for a constant or an abstract key.
   */
  private static TypeReference allocatedType(InstanceKey key) {
    if (key instanceof ConcreteTypeKey concrete) return concrete.type().getReference();
    if (key instanceof AllocationSiteInNode site) return site.concreteType().getReference();
    return null;
  }

  /**
   * The flow-graph variable of a key, as {@link ContainerElements} reads it.
   *
   * @param builder The builder.
   * @param key The key.
   * @return Its variable, or {@code null} when the propagation system holds none.
   */
  private static PointsToSetVariable variable(PropagationCallGraphBuilder builder, PointerKey key) {
    if (builder.getPropagationSystem().isImplicit(key)) return null;
    try {
      return builder.getPropagationSystem().findOrCreatePointsToSet(key);
    } catch (UnimplementedError e) {
      return null;
    }
  }
}
