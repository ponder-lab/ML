package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.ReturnValueKey;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Modeling of the Python {@code slice} builtin when called against an ndarray, as emitted by the
 * WALA Python front-end for subscript expressions of the form {@code x[:k]}. The front-end lowers
 * such a subscript into a call {@code slice(x, start, stop, step)} where {@code start} and {@code
 * step} are {@code None} and {@code stop} is the constant bound.
 *
 * <p>Scope: handles only the {@code [:k]} case &mdash; constant {@code stop}, {@code start} that is
 * {@code None} or {@code 0}, {@code step} that is {@code None} or {@code 1}. The output shape is
 * {@code [NumericDim(k), receiver[1:]&hellip;]}; the receiver's trailing dimensions are preserved
 * unchanged. For all other slice patterns ({@code [a:b]}, {@code [::s]}, non-constant bounds, etc.)
 * the generator falls back to returning the receiver's shape unchanged, matching the existing
 * passthrough behavior of the underlying {@code slice} builtin summary.
 *
 * <p>Broader slice vocabulary is out of scope per wala/ML#405.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SliceBuiltinOperation extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(SliceBuiltinOperation.class.getName());

  public SliceBuiltinOperation(PointsToSetVariable source) {
    super(source);
  }

  public SliceBuiltinOperation(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    LOGGER.fine(() -> "Entered getDefaultShapes for source=" + source);
    CallSiteView view = findCallSite(builder);
    if (view == null) {
      LOGGER.fine(() -> "No call site resolved for " + source);
      return null;
    }

    Set<List<Dimension<?>>> receiverShapes;
    try {
      // `getShapesOrSSAChain` falls back to an SSA DU walk when the PTS walk hits an implicit
      // PK (e.g., the chained `x_test.reshape(...).astype(...)` path in `neural_network.py`).
      // The SSA chain walker recognises mnist sources, astype, reshape, etc. See wala/ML#405.
      receiverShapes = getShapesOrSSAChain(builder, view.callerNode, view.receiverVn);
    } catch (IllegalArgumentException e) {
      // Both paths failed — treat as ⊤ (null) so dtype inference still proceeds.
      LOGGER.log(
          Level.FINE, "Receiver shape lookup threw IAE for receiverVn=" + view.receiverVn, e);
      receiverShapes = null;
    }
    final Set<List<Dimension<?>>> capturedReceiverShapes = receiverShapes;
    LOGGER.fine(
        () ->
            "Resolved callerNode="
                + view.callerNode
                + " receiverVn="
                + view.receiverVn
                + " receiverShapes="
                + capturedReceiverShapes);
    if (receiverShapes == null || receiverShapes.isEmpty()) return null;

    boolean startOK =
        isNone(builder, view.callerNode, view.startVn)
            || constIntEquals(builder, view.callerNode, view.startVn, 0);
    boolean stepOK =
        isNone(builder, view.callerNode, view.stepVn)
            || constIntEquals(builder, view.callerNode, view.stepVn, 1);
    Integer stop = constInt(builder, view.callerNode, view.stopVn);
    LOGGER.fine(() -> "Classified startOK=" + startOK + " stepOK=" + stepOK + " stop=" + stop);

    if (startOK && stepOK && stop != null) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (List<Dimension<?>> shape : receiverShapes) {
        if (shape == null || shape.isEmpty()) {
          return receiverShapes;
        }
        List<Dimension<?>> out = new ArrayList<>(shape.size());
        out.add(new NumericDim(stop));
        for (int i = 1; i < shape.size(); i++) out.add(shape.get(i));
        ret.add(out);
      }
      final int boundedStop = stop;
      LOGGER.fine(() -> "Matched [:k] pattern with k=" + boundedStop + " → " + ret);
      return ret;
    }

    LOGGER.fine(() -> "Non-[:k] pattern; passing receiver shape through");
    return receiverShapes;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    CallSiteView view = findCallSite(builder);
    if (view == null) return Set.of(DType.UNKNOWN);
    Set<DType> dtypes;
    try {
      // Parallel to the shape path: use the SSA-DU fallback so chained sources (mnist → astype
      // → reshape → divide → slice) still recover a concrete dtype. See wala/ML#405.
      dtypes = getDTypesOrSSAChain(builder, view.callerNode, view.receiverVn);
    } catch (IllegalArgumentException e) {
      LOGGER.log(
          Level.FINE, "Receiver dtype lookup threw IAE for receiverVn=" + view.receiverVn, e);
      dtypes = Set.of();
    }
    return dtypes.isEmpty() ? Set.of(DType.UNKNOWN) : dtypes;
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

  /**
   * Resolves the actual {@code slice(x, start, stop, step)} call site behind this generator's
   * source. Works for two source shapes:
   *
   * <ul>
   *   <li>{@link LocalPointerKey} &mdash; the caller's value number for the invoke's result. The
   *       defining instruction is the invoke; its uses carry the four args directly.
   *   <li>{@link ReturnValueKey} &mdash; the slice summary's synthetic return. We walk the call
   *       graph's predecessors of the summary CGNode and find the unique invoke targeting it. If
   *       there is more than one caller with divergent args, we bail (return {@code null}) rather
   *       than risk a wrong union.
   * </ul>
   *
   * @return A {@link CallSiteView} with the caller's node and the four arg value numbers, or {@code
   *     null} if the call site cannot be uniquely resolved.
   */
  private CallSiteView findCallSite(PropagationCallGraphBuilder builder) {
    if (source != null) {
      PointerKey pk = source.getPointerKey();
      if (pk instanceof LocalPointerKey) {
        LocalPointerKey lpk = (LocalPointerKey) pk;
        CGNode caller = lpk.getNode();
        SSAInstruction def = caller.getDU().getDef(lpk.getValueNumber());
        if (def instanceof SSAAbstractInvokeInstruction) {
          return viewOf(caller, (SSAAbstractInvokeInstruction) def);
        }
        return null;
      }
      if (pk instanceof ReturnValueKey) {
        return findViaCalleeNode(builder, ((ReturnValueKey) pk).getNode());
      }
      return null;
    }
    // Manual-node construction (from `createManualGenerator`): `getNode()` is the slice's do()
    // CGNode.
    return findViaCalleeNode(builder, getNode());
  }

  private static CallSiteView findViaCalleeNode(
      PropagationCallGraphBuilder builder, CGNode callee) {
    CallGraph cg = builder.getCallGraph();
    CallSiteView unique = null;
    for (Iterator<CGNode> it = cg.getPredNodes(callee); it.hasNext(); ) {
      CGNode caller = it.next();
      for (Iterator<com.ibm.wala.classLoader.CallSiteReference> sites =
              cg.getPossibleSites(caller, callee);
          sites.hasNext(); ) {
        com.ibm.wala.classLoader.CallSiteReference site = sites.next();
        for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(site)) {
          CallSiteView view = viewOf(caller, call);
          if (view == null) continue;
          if (unique == null) unique = view;
          else return null; // multiple call sites — ambiguous.
        }
      }
    }
    return unique;
  }

  private static CallSiteView viewOf(CGNode caller, SSAAbstractInvokeInstruction call) {
    if (call.getNumberOfUses() < 5) return null;
    return new CallSiteView(caller, call.getUse(1), call.getUse(2), call.getUse(3), call.getUse(4));
  }

  private static boolean isNone(PropagationCallGraphBuilder builder, CGNode node, int vn) {
    if (vn <= 0) return false;
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
    if (pts == null || pts.isEmpty()) return false;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return false;
      if (((ConstantKey<?>) ik).getValue() != null) return false;
    }
    return true;
  }

  private static boolean constIntEquals(
      PropagationCallGraphBuilder builder, CGNode node, int vn, int target) {
    Integer v = constInt(builder, node, vn);
    return v != null && v == target;
  }

  private static Integer constInt(PropagationCallGraphBuilder builder, CGNode node, int vn) {
    if (vn <= 0) return null;
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
    if (pts == null || pts.isEmpty()) return null;
    Integer result = null;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object v = ((ConstantKey<?>) ik).getValue();
      int n;
      if (v instanceof Integer) n = (Integer) v;
      else if (v instanceof Long) n = ((Long) v).intValue();
      else return null;
      if (result == null) result = n;
      else if (result != n) return null;
    }
    return result;
  }

  private static final class CallSiteView {
    final CGNode callerNode;
    final int receiverVn;
    final int startVn;
    final int stopVn;
    final int stepVn;

    CallSiteView(CGNode callerNode, int receiverVn, int startVn, int stopVn, int stepVn) {
      this.callerNode = callerNode;
      this.receiverVn = receiverVn;
      this.startVn = startVn;
      this.stopVn = stopVn;
      this.stepVn = stepVn;
    }
  }
}
