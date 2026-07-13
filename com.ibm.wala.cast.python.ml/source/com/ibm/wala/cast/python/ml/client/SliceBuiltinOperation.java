package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.ELLIPSIS;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.ReturnValueKey;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
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
    LOGGER.fine(() -> "Entered getDefaultShapes for source=" + describe(source));
    CallSiteView view = findCallSite(builder);
    if (view == null) {
      LOGGER.fine(() -> "No call site resolved for " + describe(source));
      return null;
    }

    Set<List<Dimension<?>>> receiverShapes;
    try {
      // `getShapesOrSSAChain` falls back to an SSA DU walk when the PTS walk hits an implicit
      // PK (e.g., the chained `x_test.reshape(...).astype(...)` path in `neural_network.py`).
      // The SSA chain walker recognises mnist sources, astype, reshape, etc. See wala/ML#405.
      receiverShapes = getShapesOrSSAChain(builder, view.callerNode(), view.receiverVn);
    } catch (IllegalArgumentException e) {
      // Both paths failed — treat as ⊤ (null) so dtype inference still proceeds.
      LOGGER.log(
          Level.FINE, "Receiver shape lookup threw IAE for receiverVn=" + view.receiverVn(), e);
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
    // A `null` receiver shape means the receiver is a tensor of unknown shape (⊤), so the slice is
    // ⊤. An empty set means the receiver is not a tensor (⊥): a subscript-slice of a non-tensor
    // (e.g. `x[1::2]` of an opaque `argparse` attribute) is not a tensor either, so propagate ⊥
    // rather than over-typing it to ⊤. wala/ML#656.
    if (receiverShapes == null) return null;
    if (receiverShapes.isEmpty()) return Set.of();

    // Multi-dim subscript: `slice(receiver, dim0, dim1, ...)` where each dimension is a
    // `slice(lower, upper, step)` object or an integer index (wala/ML#406). Distinguished from the
    // single `[:k]` form below by the presence of at least one slice-object argument.
    List<SubscriptDim> dims = parseSubscriptDims(builder, view);
    if (dims != null) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (List<Dimension<?>> shape : receiverShapes) {
        List<Dimension<?>> out = applySubscriptDims(shape, dims);
        if (out != null) ret.add(out);
      }
      if (!ret.isEmpty()) {
        LOGGER.fine(() -> "Matched multi-dim subscript " + dims + " → " + ret);
        return ret;
      }
    }

    boolean startOK =
        isNone(builder, view.callerNode(), view.startVn())
            || constIntEquals(builder, view.callerNode(), view.startVn(), 0);
    boolean stepOK =
        isNone(builder, view.callerNode(), view.stepVn())
            || constIntEquals(builder, view.callerNode(), view.stepVn(), 1);
    Integer stop = constInt(builder, view.callerNode(), view.stopVn());
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

    // The invoke didn't match the canonical `[:k]` pattern, so we don't know the exact output
    // shape. If the compound subscript includes `tf.newaxis` tokens (e.g., `x[:n, ..., newaxis]`),
    // each one inserts a size-1 dim. Append them to the receiver shape — better than dropping the
    // size-1 dim entirely, which otherwise leaks the pre-subscript shape downstream.
    int newaxisCount = countNewaxisArgs(builder, view.callerNode(), view.call());
    if (newaxisCount == 0) {
      LOGGER.fine(() -> "Non-[:k] pattern; passing receiver shape through");
      return receiverShapes;
    }
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : receiverShapes) {
      List<Dimension<?>> out = new ArrayList<>(shape);
      for (int i = 0; i < newaxisCount; i++) out.add(new NumericDim(1));
      ret.add(out);
    }
    final int capturedCount = newaxisCount;
    LOGGER.fine(() -> "Non-[:k] pattern; appended " + capturedCount + " newaxis dim(s) → " + ret);
    return ret;
  }

  /**
   * Counts how many args beyond the canonical {@code (x, start, stop, step)} resolve to a {@code
   * tf.newaxis} allocation (declared as {@code Ltensorflow/newaxis} in {@code tensorflow.xml}).
   * Used to extend the passthrough shape when the front-end lowers a compound subscript like {@code
   * x[:n, ..., tf.newaxis]} into a {@code slice(...)} invoke with extra tokens.
   *
   * @param builder The propagation call graph builder whose PA we query.
   * @param caller The caller {@link CGNode}.
   * @param call The slice invoke whose args we scan.
   * @return The number of newaxis-typed args across all use positions.
   */
  private static int countNewaxisArgs(
      PropagationCallGraphBuilder builder, CGNode caller, SSAAbstractInvokeInstruction call) {
    int count = 0;
    for (int i = 1; i < call.getNumberOfUses(); i++) {
      int useVn = call.getUse(i);
      if (useVn <= 0) continue;
      PointerKey pk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, useVn);
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
      if (pts == null || pts.isEmpty()) continue;
      for (InstanceKey ik : pts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.concreteType().getReference().equals(TensorFlowTypes.NEWAXIS)) {
          count++;
          break;
        }
      }
    }
    return count;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    CallSiteView view = findCallSite(builder);
    if (view == null) return Set.of(DType.UNKNOWN);
    Set<DType> dtypes;
    try {
      // Parallel to the shape path: use the SSA-DU fallback so chained sources (mnist → astype
      // → reshape → divide → slice) still recover a concrete dtype. See wala/ML#405.
      dtypes = getDTypesOrSSAChain(builder, view.callerNode(), view.receiverVn);
    } catch (IllegalArgumentException e) {
      LOGGER.log(
          Level.FINE, "Receiver dtype lookup threw IAE for receiverVn=" + view.receiverVn(), e);
      dtypes = Set.of();
    }
    // getDTypesOrSSAChain can return null (no dtype recovered via the SSA-DU fallback); treat it
    // like an empty set rather than NPEing on isEmpty(). See https://github.com/wala/ML/issues/602.
    return (dtypes == null || dtypes.isEmpty()) ? Set.of(DType.UNKNOWN) : dtypes;
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
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph and IR are consulted to
   *     locate the call site.
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

  /**
   * Walks the call graph's predecessors of the given callee CGNode to find the unique caller-side
   * invoke that produced it. Shared by the {@link ReturnValueKey} source case (the summary's
   * synthetic return) and the manual-node case (dispatch from {@code createManualGenerator}).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph to search.
   * @param callee The {@code Lwala/builtin/slice.do()} callee {@link CGNode}.
   * @return A {@link CallSiteView} pinned to the unique caller and its four argument value numbers,
   *     or {@code null} when no invoke resolves or when there are multiple call sites with
   *     divergent args (ambiguous &mdash; union would be wrong).
   */
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

  /**
   * Constructs a {@link CallSiteView} for a {@code slice(receiver, ...)} invoke. The single {@code
   * [:k]} form is {@code slice(receiver, start, stop, step)} (5 uses); multi-dim subscripts
   * (wala/ML#406) carry a variable number of dimension arguments. Requires only the receiver plus
   * at least one argument ({@code >= 3} uses); absent {@code start}/{@code stop}/{@code step} slots
   * are filled with {@code -1}. Returns {@code null} for a smaller use count, defending against
   * malformed invokes without throwing.
   *
   * @param caller The caller {@link CGNode} whose IR contains the invoke.
   * @param call The {@link SSAAbstractInvokeInstruction} for the slice call.
   * @return A {@link CallSiteView} with the caller and the receiver/start/stop/step value numbers
   *     (the latter three {@code -1} when absent), or {@code null} if the use count is below 3.
   */
  private static CallSiteView viewOf(CGNode caller, SSAAbstractInvokeInstruction call) {
    // use(0) is the slice builtin; use(1) is the receiver; the rest are arguments. The single
    // `[:k]` form has exactly `slice(receiver, start, stop, step)` (5 uses); multi-dim subscripts
    // (wala/ML#406) can have fewer (e.g. `x[:, 1:]` is 4 uses) or more. Require only the receiver
    // and at least one argument; absent start/stop/step slots are filled with -1.
    if (call.getNumberOfUses() < 3) return null;
    int n = call.getNumberOfUses();
    return new CallSiteView(
        caller,
        call,
        call.getUse(1),
        n > 2 ? call.getUse(2) : -1,
        n > 3 ? call.getUse(3) : -1,
        n > 4 ? call.getUse(4) : -1);
  }

  /**
   * Checks whether the points-to set of {@code vn} in {@code node} consists solely of the Python
   * {@code None} constant. Used to recognize the canonical {@code [:k]} shape where {@code start}
   * and {@code step} are absent.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param node The {@link CGNode} whose IR contains {@code vn}.
   * @param vn The SSA value number to inspect.
   * @return {@code true} iff {@code vn}'s points-to set is non-empty and every element is a {@link
   *     ConstantKey} whose value is {@code null}; {@code false} otherwise.
   */
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

  /**
   * Convenience wrapper that returns {@code true} iff {@link #constInt} resolves {@code vn} to a
   * constant integer equal to {@code target}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param node The {@link CGNode} whose IR contains {@code vn}.
   * @param vn The SSA value number to inspect.
   * @param target The integer value to compare against.
   * @return {@code true} iff {@code vn}'s resolved constant value exists and equals {@code target};
   *     {@code false} otherwise.
   */
  private static boolean constIntEquals(
      PropagationCallGraphBuilder builder, CGNode node, int vn, int target) {
    Integer v = constInt(builder, node, vn);
    return v != null && v == target;
  }

  /**
   * Resolves the points-to set of {@code vn} to a single constant integer value, accepting both
   * {@link Integer} and {@link Long} {@link ConstantKey} payloads. Returns {@code null} when the
   * set is empty, contains a non-constant, contains a non-integer constant, or contains multiple
   * distinct integer constants (ambiguity is a {@code null}, not an arbitrary pick).
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param node The {@link CGNode} whose IR contains {@code vn}.
   * @param vn The SSA value number to inspect.
   * @return The resolved constant integer, or {@code null} if the set isn't a single-valued integer
   *     constant.
   */
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

  /**
   * Parses the arguments of a {@code slice(receiver, dim0, dim1, ...)} invoke as multi-dim
   * subscript dimensions. Each argument is a {@code slice(lower, upper, step)} object (a {@code
   * :}-style dimension), an ellipsis, a newaxis, or a constant integer index (which drops an axis).
   *
   * <p>Engages multi-dim handling only when at least one argument is a slice object, which is the
   * marker that distinguishes a multi-dim subscript carrying a {@code :} from the single {@code
   * [:k]} form (whose arguments are bare {@code None}/integer bounds, not slice objects).
   * Subscripts with no slice &mdash; a pure-integer index like {@code x[0, 1]} or a pure
   * ellipsis/newaxis like {@code x[..., None]} &mdash; are lowered to a different IR form (a {@code
   * PythonPropertyRead} / {@code OBJECT_REF}) handled by {@link TensorElementGenerator} or {@link
   * NdarraySubscriptOperation}, so they do not reach this method.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param view The resolved slice call site.
   * @return The parsed dimensions, or {@code null} if there is no slice object (the single {@code
   *     [:k]} form) or any argument is unrecognized.
   */
  private List<SubscriptDim> parseSubscriptDims(
      PropagationCallGraphBuilder builder, CallSiteView view) {
    SSAAbstractInvokeInstruction call = view.call();
    CGNode caller = view.callerNode();
    int n = call.getNumberOfUses();
    List<SubscriptDim> dims = new ArrayList<>();
    boolean sawSlice = false;
    for (int u = 2; u < n; u++) {
      SubscriptDim d = classifyDim(builder, caller, call.getUse(u));
      if (d == null) return null; // bare bound (single `[:k]` form) or unrecognized.
      if (d.isSlice()) sawSlice = true;
      dims.add(d);
    }
    return sawSlice ? dims : null;
  }

  /**
   * Classifies a single subscript-argument value number as a slice dimension or an integer index.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param caller The caller {@link CGNode}.
   * @param argVn The argument value number.
   * @return a slice or index {@link SubscriptDim}, or {@code null} when the argument is neither (a
   *     bare {@code None}/non-constant — i.e. the single {@code [:k]} bound form).
   */
  private SubscriptDim classifyDim(PropagationCallGraphBuilder builder, CGNode caller, int argVn) {
    if (argVn <= 0) return null;
    if (isSliceObject(caller, argVn)) {
      SSAAbstractInvokeInstruction inv =
          (SSAAbstractInvokeInstruction) caller.getDU().getDef(argVn);
      // slice(lower, upper, step): uses 1, 2, 3.
      return SubscriptDim.slice(
          bound(builder, caller, inv.getUse(1)),
          bound(builder, caller, inv.getUse(2)),
          bound(builder, caller, inv.getUse(3)));
    }
    // Ellipsis (`...`) expands to fill the unconsumed axes; newaxis (`None` or `tf.newaxis`)
    // inserts a size-1 axis. (In a single `[:k]` slice the bare `None` bounds are not reached
    // here — that form has no slice object, so `parseSubscriptDims` discards it.)
    DimKind special = classifyEllipsisNewaxis(ptsOf(builder, caller, argVn));
    if (special == DimKind.ELLIPSIS) return SubscriptDim.ellipsis();
    if (special == DimKind.NEWAXIS) return SubscriptDim.newaxis();
    // A constant integer index drops its axis; the index value itself is not needed.
    if (constInt(builder, caller, argVn) != null) return SubscriptDim.index();
    return null;
  }

  /**
   * Classifies a subscript argument's points-to set as ellipsis ({@code ...}) or newaxis ({@code
   * None} / {@code tf.newaxis}), mirroring {@link NdarraySubscriptOperation}'s classifier.
   *
   * @param pts The argument's points-to set.
   * @return {@link DimKind#ELLIPSIS} or {@link DimKind#NEWAXIS}, or {@code null} when the argument
   *     is neither (an integer, a non-constant, an empty set, or an ambiguous mix).
   */
  private static DimKind classifyEllipsisNewaxis(OrdinalSet<InstanceKey> pts) {
    if (pts == null || pts.isEmpty()) return null;
    boolean ellipsis = false;
    boolean newaxis = false;
    for (InstanceKey ik : pts) {
      if (ik instanceof ConstantKey) {
        Object v = ((ConstantKey<?>) ik).getValue();
        if (ELLIPSIS.equals(v)) ellipsis = true;
        else if (v == null) newaxis = true; // `None` is `np.newaxis`.
        else return null; // integer/other — not special.
      } else {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.concreteType().getReference().equals(TensorFlowTypes.NEWAXIS))
          newaxis = true;
        else return null;
      }
    }
    if (ellipsis && newaxis) return null; // ambiguous.
    return ellipsis ? DimKind.ELLIPSIS : newaxis ? DimKind.NEWAXIS : null;
  }

  /**
   * Returns the points-to set of {@code vn} in {@code caller}, or {@code null} if {@code vn} is
   * invalid.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param caller The caller {@link CGNode}.
   * @param vn The value number.
   * @return The points-to set, or {@code null}.
   */
  private static OrdinalSet<InstanceKey> ptsOf(
      PropagationCallGraphBuilder builder, CGNode caller, int vn) {
    if (vn <= 0) return null;
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, vn);
    return builder.getPointerAnalysis().getPointsToSet(pk);
  }

  /**
   * Returns whether {@code argVn}'s defining instruction is a {@code slice(...)} object
   * construction (an invoke whose callee is the {@link PythonTypes#SLICE_BUILTIN} allocation).
   *
   * @param caller The caller {@link CGNode}.
   * @param argVn The argument value number.
   * @return {@code true} iff {@code argVn} is defined by a slice-object construction.
   */
  private static boolean isSliceObject(CGNode caller, int argVn) {
    SSAInstruction def = caller.getDU().getDef(argVn);
    if (!(def instanceof SSAAbstractInvokeInstruction)) return false;
    SSAAbstractInvokeInstruction inv = (SSAAbstractInvokeInstruction) def;
    if (inv.getNumberOfUses() < 4) return false;
    SSAInstruction funcDef = caller.getDU().getDef(inv.getUse(0));
    return funcDef instanceof SSANewInstruction
        && ((SSANewInstruction) funcDef)
            .getNewSite()
            .getDeclaredType()
            .equals(PythonTypes.SLICE_BUILTIN);
  }

  /**
   * Resolves a slice bound ({@code lower}, {@code upper}, or {@code step}) value number to a {@link
   * Bound}: a constant integer, {@code None}, or unknown (non-constant).
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param caller The caller {@link CGNode}.
   * @param vn The bound's value number.
   * @return The resolved {@link Bound}.
   */
  private static Bound bound(PropagationCallGraphBuilder builder, CGNode caller, int vn) {
    if (isNone(builder, caller, vn)) return Bound.NONE;
    Integer v = constInt(builder, caller, vn);
    return v != null ? Bound.of(v) : Bound.UNKNOWN;
  }

  /**
   * Applies the parsed subscript dimensions to a single receiver shape. Walks the receiver's axes
   * in order: a slice dimension maps its axis to the sliced extent; an index dimension drops its
   * axis. Any receiver axes beyond the explicit dimensions are preserved (an implicit trailing
   * {@code :}).
   *
   * @param input The receiver's shape dims, in order.
   * @param dims The parsed subscript dimensions, in order.
   * @return The output shape, or {@code null} when there are more dimensions than receiver axes.
   */
  private static List<Dimension<?>> applySubscriptDims(
      List<Dimension<?>> input, List<SubscriptDim> dims) {
    // A slice or index consumes one receiver axis; newaxis and ellipsis do not. An ellipsis
    // expands to the axes left over after the consuming dimensions; absent an ellipsis, those
    // left-over axes form an implicit trailing `:`.
    int consuming = 0;
    int ellipsisCount = 0;
    for (SubscriptDim d : dims) {
      DimKind k = d.kind();
      if (k == DimKind.SLICE || k == DimKind.INDEX) consuming++;
      else if (k == DimKind.ELLIPSIS) ellipsisCount++;
    }
    if (ellipsisCount > 1 || consuming > input.size()) return null;
    int ellipsisFill = input.size() - consuming;

    List<Dimension<?>> out = new ArrayList<>();
    int axis = 0;
    for (SubscriptDim d : dims) {
      switch (d.kind()) {
        case NEWAXIS:
          out.add(new NumericDim(1));
          break;
        case ELLIPSIS:
          for (int k = 0; k < ellipsisFill; k++) out.add(input.get(axis++));
          break;
        case SLICE:
          out.add(sliceExtent(input.get(axis++), d));
          break;
        case INDEX:
          axis++; // drop the axis.
          break;
      }
    }
    for (; axis < input.size(); axis++) out.add(input.get(axis)); // implicit trailing `:`.
    return out;
  }

  /**
   * Computes the output extent of one axis under a slice dimension, per Python slice semantics with
   * a unit (or absent) step. A full slice ({@code :}) preserves the axis. Constant bounds against a
   * numeric axis compute the extent numerically (negative bounds index from the end); a {@code :k}
   * slice against an unknown axis still yields {@code k}. Anything else degrades — to {@link
   * DynamicDim} when the receiver's axis is itself {@code None}-evidenced, else to {@link
   * UnresolvedDim} (wala/ML#721) — never to ⊤; the axis stays present but its extent is unknown.
   *
   * @param recv The receiver's dimension for this axis.
   * @param d The slice dimension.
   * @return The output dimension for this axis.
   */
  private static Dimension<?> sliceExtent(Dimension<?> recv, SubscriptDim d) {
    Bound lower = d.lower(), upper = d.upper(), step = d.step();
    if (lower.isNone() && upper.isNone() && step.isNone()) return recv; // `:`
    // Slice bounds and steps are Python scalars, so an uncomputable extent over a fixed axis is a
    // fixed runtime size the analysis could not compute; only slicing a `None` axis yields a
    // `None` extent (wala/ML#721).
    Dimension<?> degraded =
        recv instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE;
    int s = step.isNone() ? 1 : (step.isInt() ? step.value() : 0);
    if (s != 1) return degraded; // non-unit/non-constant/negative step.
    Integer n = (recv instanceof NumericDim) ? ((NumericDim) recv).value() : null;
    if (n == null) {
      // Unknown axis: only `:k` (lower None, constant non-negative upper) is computable.
      if (lower.isNone() && upper.isInt() && upper.value() >= 0)
        return new NumericDim(upper.value());
      return degraded;
    }
    if ((!lower.isNone() && !lower.isInt()) || (!upper.isNone() && !upper.isInt()))
      return degraded; // non-constant bound against a known axis.
    int lo = lower.isNone() ? 0 : lower.value();
    int hi = upper.isNone() ? n : upper.value();
    if (lo < 0) lo = Math.max(0, n + lo);
    if (hi < 0) hi = Math.max(0, n + hi);
    lo = Math.min(lo, n);
    hi = Math.min(hi, n);
    return new NumericDim(Math.max(0, hi - lo));
  }

  /**
   * A resolved slice bound: a constant integer ({@code value} non-null), {@code None} ({@code
   * none}), or unknown/non-constant (neither).
   */
  private record Bound(Integer value, boolean none) {
    static final Bound NONE = new Bound(null, true);
    static final Bound UNKNOWN = new Bound(null, false);

    static Bound of(int v) {
      return new Bound(v, false);
    }

    boolean isNone() {
      return none;
    }

    boolean isInt() {
      return value != null;
    }
  }

  /** The kind of a multi-dim subscript dimension. */
  private enum DimKind {
    SLICE,
    INDEX,
    NEWAXIS,
    ELLIPSIS
  }

  /**
   * One dimension of a multi-dim subscript: a slice (carrying {@code lower}/{@code upper}/{@code
   * step} bounds), an integer index, a newaxis, or an ellipsis.
   */
  private record SubscriptDim(DimKind kind, Bound lower, Bound upper, Bound step) {
    static SubscriptDim slice(Bound lower, Bound upper, Bound step) {
      return new SubscriptDim(DimKind.SLICE, lower, upper, step);
    }

    static SubscriptDim index() {
      return new SubscriptDim(DimKind.INDEX, null, null, null);
    }

    static SubscriptDim newaxis() {
      return new SubscriptDim(DimKind.NEWAXIS, null, null, null);
    }

    static SubscriptDim ellipsis() {
      return new SubscriptDim(DimKind.ELLIPSIS, null, null, null);
    }

    boolean isSlice() {
      return kind == DimKind.SLICE;
    }
  }

  /**
   * A positional snapshot of a {@code slice(x, start, stop, step)} invoke, pinned to the caller
   * whose IR contains it. Bridges the gap between a generator source (either a {@link
   * LocalPointerKey} in the caller or a {@link ReturnValueKey} from the slice summary) and the
   * value numbers of the four actual args, so {@link #getDefaultShapes} and {@link
   * #getDefaultDTypes} can resolve them against the caller's SSA.
   *
   * @param callerNode The caller {@link CGNode} whose IR contains the invoke.
   * @param call The {@link SSAAbstractInvokeInstruction} itself, kept so callers can scan args
   *     beyond the canonical four for compound-subscript tokens (e.g., {@code tf.newaxis}).
   * @param receiverVn SSA value number of the receiver ({@code x}) in {@code callerNode}.
   * @param startVn SSA value number of the {@code start} argument.
   * @param stopVn SSA value number of the {@code stop} argument.
   * @param stepVn SSA value number of the {@code step} argument.
   */
  private record CallSiteView(
      CGNode callerNode,
      SSAAbstractInvokeInstruction call,
      int receiverVn,
      int startVn,
      int stopVn,
      int stepVn) {}
}
