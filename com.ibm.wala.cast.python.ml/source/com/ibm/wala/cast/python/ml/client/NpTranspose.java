package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code np.transpose(a, axes)} and the {@code ndarray.transpose(axes)} method form
 * (wala/ML#835): the output permutes the input's dimensions by the constant {@code axes},
 * defaulting to full reversal when {@code axes} is absent or {@code None}; the dtype passes through
 * from the input. An unresolvable {@code axes} still knows the RANK, since a permutation preserves
 * it, so each member degrades to an unresolved-per-axis shape of its own rank rather than ⊤.
 *
 * <p>The two forms differ in argument geometry, so the form is fixed at construction: the function
 * form binds the input at position 0 ({@code a}) and {@code axes} at position 1, while the method
 * form binds {@code axes} at position 0 and takes its input from the RECEIVER, which the trampoline
 * never passes as an argument — it sits behind the invoke's function value and is read off the
 * property read that defined it (the {@code tolist} precedent, wala/ML#796), through the caller
 * walk when this anchor's node is the summary itself.
 */
public class NpTranspose extends PassThroughUnaryTensorGenerator {

  /** How the {@code axes} argument resolved. */
  private enum AxesKind {
    /** Absent, or every member is the {@code None} constant: the reversal default. */
    ABSENT_OR_NONE,
    /** Exactly one constant permutation. */
    CONSTANT,
    /** One constant permutation beside a live {@code None}: both outcomes join. */
    MIXED_WITH_NONE,
    /** Present but not resolvable to one constant permutation. */
    UNRESOLVABLE
  }

  /** An {@code axes} resolution: its kind, and the permutation for the constant-bearing kinds. */
  private record AxesResolution(AxesKind kind, List<Integer> axes) {}

  /** Whether this instance serves the {@code ndarray.transpose} method form. */
  private final boolean methodForm;

  public NpTranspose(PointsToSetVariable source, boolean methodForm) {
    super(source);
    this.methodForm = methodForm;
  }

  public NpTranspose(CGNode node, boolean methodForm) {
    super(node);
    this.methodForm = methodForm;
  }

  @Override
  protected int getInputParameterPosition() {
    return this.methodForm ? RECEIVER_PARAMETER_POSITION : 0;
  }

  @Override
  protected String getInputParameterName() {
    return this.methodForm ? SELF : "a";
  }

  /** The position of the {@code axes} argument in this form's frame, {@code self} excluded. */
  private int getAxesPosition() {
    return this.methodForm ? 0 : 1;
  }

  /**
   * This generator transforms its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682, the {@code Transpose} precedent).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    ShapeResult result = this.getDefaultShapeResult(builder);
    return result.isPartial() ? result.members() : result.toLegacy();
  }

  /**
   * Member-wise record view (wala/ML#718): each input member permutes independently, and the
   * input's unknown remainder rides through as the result's remainder.
   *
   * @param builder The propagation call graph builder.
   * @return The composed result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    ShapeResult input = this.inputShapes(builder);
    if (input.members().isEmpty()) return ShapeResult.unknown();

    AxesResolution axes = this.resolveAxes(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : input.members()) {
      switch (axes.kind()) {
        case UNRESOLVABLE -> {
          // The permutation is unknown, but a permutation preserves the rank: an
          // unresolved-per-axis shape of the member's own rank, not ⊤.
          List<Dimension<?>> out = new ArrayList<>();
          for (int i = 0; i < shape.size(); i++) out.add(UnresolvedDim.INSTANCE);
          ret.add(out);
        }
        case ABSENT_OR_NONE -> ret.add(reversed(shape));
        case CONSTANT -> {
          List<Dimension<?>> out = this.permuted(shape, axes.axes());
          if (out != null) ret.add(out);
        }
        case MIXED_WITH_NONE -> {
          // Both branches are live in the points-to set, so both outcomes join (a mixed
          // `axes = None if cond else (2, 0, 1)` must not drop the reversal member).
          ret.add(reversed(shape));
          List<Dimension<?>> out = this.permuted(shape, axes.axes());
          if (out != null) ret.add(out);
        }
      }
    }
    return ret.isEmpty() ? ShapeResult.unknown() : new ShapeResult(ret, input.hasUnknown());
  }

  /**
   * Reverses a shape: the {@code axes}-absent default.
   *
   * @param shape The input member.
   * @return The reversed shape.
   */
  private static List<Dimension<?>> reversed(List<Dimension<?>> shape) {
    List<Dimension<?>> out = new ArrayList<>(shape);
    Collections.reverse(out);
    return out;
  }

  /**
   * Applies a constant permutation to one member, sharing {@link Transpose#permuteShape} for the
   * bijectivity check ("anything else is unsound") after normalizing numpy's negative axes, which
   * count from the end. A member the permutation cannot apply to — rank mismatch, out-of-range, or
   * repeated axes — is a guaranteed run-time error for this call, so the {@code null} return is
   * infeasible-path pruning, not an unmarked remainder.
   *
   * @param shape The input member.
   * @param axes The constant permutation, possibly with negative entries.
   * @return The permuted shape, or {@code null} when the permutation cannot apply.
   */
  private List<Dimension<?>> permuted(List<Dimension<?>> shape, List<Integer> axes) {
    List<Integer> normalized = new ArrayList<>(axes.size());
    for (int axis : axes) normalized.add(axis < 0 ? axis + shape.size() : axis);
    return Transpose.permuteShape(shape, normalized);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The dtype passes through from the input: the function form's argument slot, or the
   *     method form's receiver.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    if (!this.methodForm) {
      int inputVn = this.inputValueNumber(builder);
      if (inputVn > 0) {
        Set<DType> dTypes = getDTypesOrSSAChain(builder, this.getNode(), inputVn);
        if (dTypes != null && !dTypes.isEmpty()) return dTypes;
      }
      return super.getDefaultDTypes(builder);
    }
    Set<DType> viaReceiver = EnumSet.noneOf(DType.class);
    for (Pair<CGNode, Integer> receiver : this.receiverValueNumbers(builder)) {
      try {
        Set<DType> dTypes = getDTypesOrSSAChain(builder, receiver.fst, receiver.snd);
        if (dTypes != null) viaReceiver.addAll(dTypes);
      } catch (IllegalArgumentException e) {
        // This receiver candidate does not resolve; others may.
      }
    }
    if (!viaReceiver.isEmpty()) return viaReceiver;
    return super.getDefaultDTypes(builder);
  }

  /**
   * Resolves the input's shapes: the function form's argument slot, or the method form's receiver.
   *
   * @param builder The propagation call graph builder.
   * @return The input's resolution; unknown (⊤) when the route does not resolve — the input is not
   *     thereby proven a non-tensor.
   */
  private ShapeResult inputShapes(PropagationCallGraphBuilder builder) {
    if (!this.methodForm) {
      int inputVn = this.inputValueNumber(builder);
      if (inputVn > 0) {
        ShapeResult viaArgument = this.getShapeResult(builder, this.getNode(), inputVn, true);
        if (!viaArgument.members().isEmpty()) return viaArgument;
      }
      return ShapeResult.unknown();
    }
    Set<List<Dimension<?>>> viaReceiver = HashSetFactory.make();
    for (Pair<CGNode, Integer> receiver : this.receiverValueNumbers(builder)) {
      try {
        Set<List<Dimension<?>>> shapes = getShapesOrSSAChain(builder, receiver.fst, receiver.snd);
        if (shapes != null) viaReceiver.addAll(shapes);
      } catch (IllegalArgumentException e) {
        // This receiver candidate does not resolve; others may.
      }
    }
    return viaReceiver.isEmpty() ? ShapeResult.unknown() : ShapeResult.of(viaReceiver);
  }

  /**
   * The method form's receiver candidates: the {@code x} of {@code x.transpose(...)}, read off the
   * property read that defined the invoke's function object (the {@code tolist} precedent,
   * wala/ML#796) — locally when this anchor carries the invoke, and through the caller walk when
   * this anchor's node is the summary itself (a delegation anchor has no invoke in its own frame).
   *
   * @param builder The propagation call graph builder.
   * @return Pairs of the frame holding the receiver and its value number there.
   */
  private List<Pair<CGNode, Integer>> receiverValueNumbers(PropagationCallGraphBuilder builder) {
    List<Pair<CGNode, Integer>> ret = new ArrayList<>();
    SSAAbstractInvokeInstruction localInvoke = this.getInvokeInstruction();
    if (localInvoke != null) {
      CGNode node = this.getNode();
      if (node.getDU() != null) {
        SSAInstruction funcDef = node.getDU().getDef(localInvoke.getUse(0));
        if (funcDef instanceof PythonPropertyRead read)
          ret.add(Pair.make(node, read.getObjectRef()));
      }
      return ret;
    }
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (caller.getDU() == null) continue;
      SSAInstruction funcDef = caller.getDU().getDef(callerInvoke.snd.getUse(0));
      if (funcDef instanceof PythonPropertyRead read)
        ret.add(Pair.make(caller, read.getObjectRef()));
    }
    return ret;
  }

  /**
   * The input's value number in the anchoring frame, for the function form's argument slot.
   *
   * @param builder The propagation call graph builder.
   * @return The input's value number, or {@code -1} when unavailable.
   */
  private int inputValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getInputParameterPosition(), this.getInputParameterName(), true);
  }

  /**
   * Classifies the {@code axes} argument in one pass over its points-to set: the {@code None}
   * members and the folded permutation candidates are read together, so a mixed set keeps both
   * outcomes and the None-selects-reversal rule lives in exactly one place.
   *
   * @param builder The propagation call graph builder.
   * @return The resolution.
   */
  private AxesResolution resolveAxes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> axesPts =
        this.getArgumentPointsToSet(builder, this.getAxesPosition(), "axes");
    if (axesPts == null || axesPts.isEmpty())
      return new AxesResolution(AxesKind.ABSENT_OR_NONE, null);

    boolean sawNone = false;
    boolean sawUnresolvable = false;
    Set<List<Integer>> candidates = HashSetFactory.make();

    for (InstanceKey instanceKey : axesPts) {
      if (instanceKey instanceof ConstantKey<?> constant && constant.getValue() == null) {
        sawNone = true;
        continue;
      }
      Set<List<Dimension<?>>> folded;
      try {
        folded = this.getShapesFromShapeArgument(builder, Collections.singleton(instanceKey));
      } catch (IllegalStateException e) {
        // An unrecognized axes form: present but unresolvable.
        sawUnresolvable = true;
        continue;
      }
      if (folded == null) {
        sawUnresolvable = true;
        continue;
      }
      for (List<Dimension<?>> fold : folded) {
        List<Integer> candidate = new ArrayList<>(fold.size());
        boolean numeric = true;
        for (Dimension<?> d : fold) {
          if (!(d instanceof NumericDim)) {
            numeric = false;
            break;
          }
          candidate.add(((NumericDim) d).value());
        }
        if (numeric) candidates.add(candidate);
        else sawUnresolvable = true;
      }
    }

    if (sawUnresolvable || candidates.size() > 1)
      return new AxesResolution(AxesKind.UNRESOLVABLE, null);
    if (candidates.isEmpty()) return new AxesResolution(AxesKind.ABSENT_OR_NONE, null);
    return new AxesResolution(
        sawNone ? AxesKind.MIXED_WITH_NONE : AxesKind.CONSTANT, candidates.iterator().next());
  }
}
