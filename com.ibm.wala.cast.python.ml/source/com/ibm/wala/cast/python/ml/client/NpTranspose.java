package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code np.transpose(a, axes)} and the {@code ndarray.transpose(...)} method form
 * (wala/ML#835): the output permutes the input's dimensions by the constant {@code axes},
 * defaulting to full reversal when {@code axes} is absent or {@code None}; the dtype passes through
 * from the input. An unresolvable {@code axes} still knows the RANK, since a permutation preserves
 * it, so each member degrades to an unresolved-per-axis shape of its own rank rather than ⊤.
 *
 * <p>Both modeled forms place the input at position 0 after {@code self} (the function form's
 * {@code a}, the method form's trampoline-supplied receiver), so one generator serves both dispatch
 * registrations.
 */
public class NpTranspose extends PassThroughUnaryTensorGenerator {

  /** The position of the {@code axes} argument, {@code self} excluded. */
  private static final int AXES_POSITION = 1;

  public NpTranspose(PointsToSetVariable source) {
    super(source);
  }

  public NpTranspose(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "a";
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

    List<Integer> axes = this.getConstantAxes(builder);
    boolean axesUnresolvable = this.axesPresentButUnresolvable(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : input.members()) {
      if (axesUnresolvable) {
        // The permutation is unknown, but a permutation preserves the rank: an
        // unresolved-per-axis shape of the member's own rank, not ⊤.
        List<Dimension<?>> out = new ArrayList<>();
        for (int i = 0; i < shape.size(); i++) out.add(UnresolvedDim.INSTANCE);
        ret.add(out);
        continue;
      }
      if (axes == null) {
        // Absent or `None`: full reversal, the API default.
        List<Dimension<?>> out = new ArrayList<>(shape);
        Collections.reverse(out);
        ret.add(out);
        continue;
      }
      // A constant permutation applies only to members of the matching rank; a member the
      // permutation cannot apply to is a guaranteed run-time error for this call, so the skip is
      // infeasible-path pruning, not an unmarked remainder.
      if (axes.size() != shape.size()) continue;
      List<Dimension<?>> out = new ArrayList<>();
      boolean valid = true;
      for (int axis : axes) {
        if (axis < 0) axis += shape.size();
        if (axis < 0 || axis >= shape.size()) {
          valid = false;
          break;
        }
        out.add(shape.get(axis));
      }
      if (valid) ret.add(out);
    }
    return ret.isEmpty() ? ShapeResult.unknown() : new ShapeResult(ret, input.hasUnknown());
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The dtype passes through from the input, resolved by the same value number the shape
   *     path uses, so the method form's trampoline-supplied receiver serves both.
   */
  @Override
  protected Set<com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType> getDefaultDTypes(
      PropagationCallGraphBuilder builder) {
    int inputVn = this.inputValueNumber(builder);
    if (inputVn > 0) {
      Set<com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType> dTypes =
          getDTypesOrSSAChain(builder, this.getNode(), inputVn);
      if (dTypes != null && !dTypes.isEmpty()) return dTypes;
    }
    // The method form binds no argument slot: the receiver sits behind each caller's invoke
    // function value, so it is read off the property read in the caller's frame (the `tolist`
    // precedent, wala/ML#796, lifted through the caller walk since this anchor's node is the
    // summary).
    Set<com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType> viaReceiver =
        java.util.EnumSet.noneOf(com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.class);
    for (com.ibm.wala.util.collections.Pair<CGNode, com.ibm.wala.ssa.SSAAbstractInvokeInstruction>
        callerInvoke : getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (caller.getDU() == null) continue;
      com.ibm.wala.ssa.SSAInstruction funcDef = caller.getDU().getDef(callerInvoke.snd.getUse(0));
      if (!(funcDef instanceof com.ibm.wala.cast.python.ssa.PythonPropertyRead read)) continue;
      try {
        Set<com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType> dTypes =
            getDTypesOrSSAChain(builder, caller, read.getObjectRef());
        if (dTypes != null) viaReceiver.addAll(dTypes);
      } catch (IllegalArgumentException e) {
        // This caller does not resolve; others may.
      }
    }
    if (!viaReceiver.isEmpty()) return viaReceiver;
    return super.getDefaultDTypes(builder);
  }

  /**
   * Resolves the input's shapes: the argument route first, which the function form binds, then the
   * receiver route for the method form, whose trampoline supplies the receiver behind the invoke's
   * function value rather than in an argument slot (the `tolist` precedent, wala/ML#796).
   *
   * @param builder The propagation call graph builder.
   * @return The input's resolution; ⊥-shaped when neither route resolves.
   */
  private ShapeResult inputShapes(PropagationCallGraphBuilder builder) {
    int inputVn = this.inputValueNumber(builder);
    if (inputVn > 0) {
      ShapeResult viaArgument = this.getShapeResult(builder, this.getNode(), inputVn, true);
      if (!viaArgument.members().isEmpty()) return viaArgument;
    }
    Set<List<Dimension<?>>> viaReceiver = HashSetFactory.make();
    for (com.ibm.wala.util.collections.Pair<CGNode, com.ibm.wala.ssa.SSAAbstractInvokeInstruction>
        callerInvoke : getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (caller.getDU() == null) continue;
      com.ibm.wala.ssa.SSAInstruction funcDef = caller.getDU().getDef(callerInvoke.snd.getUse(0));
      if (!(funcDef instanceof com.ibm.wala.cast.python.ssa.PythonPropertyRead read)) continue;
      try {
        Set<List<Dimension<?>>> shapes = getShapesOrSSAChain(builder, caller, read.getObjectRef());
        if (shapes != null) viaReceiver.addAll(shapes);
      } catch (IllegalArgumentException e) {
        // This caller does not resolve; others may.
      }
    }
    return viaReceiver.isEmpty() ? ShapeResult.bottom() : ShapeResult.of(viaReceiver);
  }

  /**
   * The input's value number in the anchoring frame. The function form supplies it as the first
   * positional argument; the method form supplies the receiver through the trampoline, invisible at
   * the user call site, so the fallback reads the {@code x} of {@code x.transpose(...)} off the
   * property read that defined the invoke's function object (the {@code tolist} precedent,
   * wala/ML#796).
   *
   * @param builder The propagation call graph builder.
   * @return The input's value number, or {@code -1} when unavailable.
   */
  private int inputValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getInputParameterPosition(), this.getInputParameterName(), true);
  }

  /**
   * Resolves the constant {@code axes} permutation.
   *
   * @param builder The propagation call graph builder.
   * @return The permutation's integer entries, or {@code null} when {@code axes} is absent or
   *     {@code None} (the reversal default applies).
   */
  private List<Integer> getConstantAxes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> axesPts = this.getAxesPointsToSet(builder);
    if (axesPts == null || axesPts.isEmpty()) return null;

    Set<List<Dimension<?>>> axesShapes = this.getShapesFromShapeArgument(builder, axesPts);
    if (axesShapes == null || axesShapes.size() != 1) return null;

    List<Integer> axes = new ArrayList<>();
    for (Dimension<?> d : axesShapes.iterator().next()) {
      if (!(d.value() instanceof Integer)) return null;
      axes.add((Integer) d.value());
    }
    return axes.isEmpty() ? null : axes;
  }

  /**
   * Whether {@code axes} is supplied but does not resolve to one constant integer permutation.
   *
   * @param builder The propagation call graph builder.
   * @return {@code true} iff the argument is present and unresolvable.
   */
  private boolean axesPresentButUnresolvable(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> axesPts = this.getAxesPointsToSet(builder);
    if (axesPts == null || axesPts.isEmpty()) return false;
    // `None` explicitly selects the reversal default, which is resolvable.
    Set<Object> constants = getConstantValues(axesPts, false);
    if (constants != null && constants.size() == 1 && constants.iterator().next() == null)
      return false;
    return this.getConstantAxes(builder) == null;
  }

  /**
   * The {@code axes} argument's points-to set.
   *
   * @param builder The propagation call graph builder.
   * @return The points-to set, or {@code null} when the argument is absent.
   */
  private OrdinalSet<InstanceKey> getAxesPointsToSet(PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(builder, AXES_POSITION, "axes");
  }
}
