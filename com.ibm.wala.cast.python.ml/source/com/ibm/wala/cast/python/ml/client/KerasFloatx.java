package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SET_FLOATX;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.logging.Logger;

/**
 * The values {@code tf.keras.backend.floatx()} may return where a node runs (<a
 * href="https://github.com/wala/ML/issues/871">wala/ML#871</a>). It is {@code float32} unless a
 * reachable {@code tf.keras.backend.set_floatx} call may have changed it, which a Keras API reads
 * wherever its dtype defaults to {@code floatx()}: an {@code Input} without a {@code dtype}, a
 * layer's compute dtype, its weights.
 *
 * <p>A {@code set_floatx} call's effect is the whole program unless the function making it restores
 * the default on every normal path from the call to its exit, by a later {@code
 * set_floatx("float32")} in the same function. Then its effect is scoped: the function itself and
 * everything reachable from the calls between the two. The scoping assumes the function completes
 * normally; one that raises before the restore leaves the program on the value it set. A test
 * runner runs every test in one process, so a setting at a module's top level, which no function
 * restores, changes the run for every entry point, which the whole-program effect models.
 *
 * <p>The query is flow-insensitive within its scope: a node it reaches may run before the call as
 * well as after, so the default is always among the values.
 */
public final class KerasFloatx {

  private static final Logger LOGGER = Logger.getLogger(KerasFloatx.class.getName());

  /** The default value of {@code floatx()}. */
  private static final Set<DType> DEFAULT = Collections.unmodifiableSet(EnumSet.of(DType.FLOAT32));

  /** Per builder, its answers. */
  private static final Map<PropagationCallGraphBuilder, KerasFloatx> CACHE =
      Collections.synchronizedMap(new WeakHashMap<>());

  /** The values every node may see. */
  private final Set<DType> global;

  /** The values each scoped call may set, with the nodes its effect reaches. */
  private final List<Pair<Set<CGNode>, Set<DType>>> scoped;

  private KerasFloatx(Set<DType> global, List<Pair<Set<CGNode>, Set<DType>>> scoped) {
    this.global = global;
    this.scoped = scoped;
  }

  /**
   * The values {@code floatx()} may return where the given node runs.
   *
   * @param builder The builder whose call graph and pointer analysis are final.
   * @param node The node, or {@code null} when it is not known, for the values any node may see.
   * @return The dtypes, {@code float32} among them; {@link DType#UNKNOWN} among them where a call
   *     sets a value the analysis cannot read.
   */
  public static Set<DType> at(PropagationCallGraphBuilder builder, CGNode node) {
    KerasFloatx floatx;
    synchronized (CACHE) {
      floatx = CACHE.computeIfAbsent(builder, KerasFloatx::compute);
    }
    if (floatx.global == DEFAULT && floatx.scoped.isEmpty()) return DEFAULT;
    Set<DType> ret = EnumSet.copyOf(floatx.global);
    for (Pair<Set<CGNode>, Set<DType>> s : floatx.scoped)
      if (node == null || s.fst.contains(node)) ret.addAll(s.snd);
    return ret;
  }

  /**
   * Clears a builder's answers, whose nodes keep the weak key reachable.
   *
   * @param builder The builder.
   */
  static void clearCache(PropagationCallGraphBuilder builder) {
    CACHE.remove(builder);
  }

  private static KerasFloatx compute(PropagationCallGraphBuilder builder) {
    CallGraph callGraph = builder.getCallGraph();
    Set<CGNode> setters = callGraph.getNodes(SET_FLOATX);
    if (setters.isEmpty()) return new KerasFloatx(DEFAULT, List.of());

    Set<DType> global = EnumSet.copyOf(DEFAULT);
    List<Pair<Set<CGNode>, Set<DType>>> scoped = new ArrayList<>();

    for (CGNode setter : setters)
      for (CGNode caller : (Iterable<CGNode>) () -> callGraph.getPredNodes(setter)) {
        IR ir = caller.getIR();
        List<PythonInvokeInstruction> calls = setFloatxCalls(callGraph, caller, setters);
        if (ir == null || calls.isEmpty()) {
          // A caller whose calls cannot be read, a synthetic one among them: its value is unknown
          // and its effect is everywhere.
          global.add(DType.UNKNOWN);
          continue;
        }
        for (PythonInvokeInstruction call : calls) {
          Set<DType> values = values(builder, caller, call);
          Set<CGNode> region = restoredRegion(callGraph, caller, ir, call, calls, builder);
          if (region == null) {
            global.addAll(values);
            LOGGER.fine(
                () ->
                    "wala/ML#871 set_floatx in "
                        + caller.getMethod().getSignature()
                        + " is unrestored: "
                        + values
                        + " everywhere.");
          } else {
            scoped.add(Pair.make(region, values));
            LOGGER.fine(
                () ->
                    "wala/ML#871 set_floatx in "
                        + caller.getMethod().getSignature()
                        + " is restored: "
                        + values
                        + " in "
                        + region.size()
                        + " nodes.");
          }
        }
      }

    return new KerasFloatx(global, scoped);
  }

  /** The calls in {@code caller} that may target a {@code set_floatx} node. */
  private static List<PythonInvokeInstruction> setFloatxCalls(
      CallGraph callGraph, CGNode caller, Set<CGNode> setters) {
    List<PythonInvokeInstruction> ret = new ArrayList<>();
    IR ir = caller.getIR();
    if (ir == null) return ret;
    for (CGNode setter : setters)
      for (var it = callGraph.getPossibleSites(caller, setter); it.hasNext(); )
        for (SSAAbstractInvokeInstruction call : ir.getCalls(it.next()))
          if (call instanceof PythonInvokeInstruction pythonCall && !ret.contains(pythonCall))
            ret.add(pythonCall);
    return ret;
  }

  /**
   * The values a {@code set_floatx} call may set: each string it may be passed, as a dtype, and
   * {@link DType#UNKNOWN} for anything else, an unreadable argument among them.
   */
  private static Set<DType> values(
      PropagationCallGraphBuilder builder, CGNode caller, PythonInvokeInstruction call) {
    int vn = call.getNumberOfPositionalParameters() > 1 ? call.getUse(1) : call.getUse("value");
    Set<DType> ret = EnumSet.noneOf(DType.class);
    if (vn < 0) {
      ret.add(DType.UNKNOWN);
      return ret;
    }
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    OrdinalSet<InstanceKey> pointsToSet =
        pointerAnalysis.getPointsToSet(
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(caller, vn));
    if (pointsToSet == null || pointsToSet.isEmpty()) {
      ret.add(DType.UNKNOWN);
      return ret;
    }
    for (InstanceKey key : pointsToSet) {
      DType dtype = null;
      if (key instanceof ConstantKey<?> constant && constant.getValue() instanceof String name)
        try {
          dtype = DType.valueOf(name.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
          // Not a dtype name; `set_floatx` raises on it, and it reads as unknown below.
        }
      ret.add(dtype != null && dtype.isFloatingPoint() ? dtype : DType.UNKNOWN);
    }
    return ret;
  }

  /**
   * The nodes a {@code set_floatx} call's effect reaches when its function restores the default on
   * every normal path from it to the exit, or {@code null} when none does.
   *
   * @param callGraph The call graph.
   * @param caller The node making the call.
   * @param ir The caller's IR.
   * @param call The call.
   * @param calls The caller's {@code set_floatx} calls, among which a restore is sought.
   * @param builder The builder, for reading a restore's argument.
   * @return The caller and the nodes reachable from its calls between {@code call} and a restore,
   *     or {@code null}.
   */
  private static Set<CGNode> restoredRegion(
      CallGraph callGraph,
      CGNode caller,
      IR ir,
      PythonInvokeInstruction call,
      List<PythonInvokeInstruction> calls,
      PropagationCallGraphBuilder builder) {
    SSACFG cfg = ir.getControlFlowGraph();
    for (PythonInvokeInstruction restore : calls) {
      if (restore == call || !values(builder, caller, restore).equals(DEFAULT)) continue;
      List<SSAAbstractInvokeInstruction> between = callsBetween(cfg, ir, call, restore);
      if (between == null) continue;
      Set<CGNode> region = HashSetFactory.make();
      Deque<CGNode> worklist = new ArrayDeque<>();
      region.add(caller);
      for (SSAAbstractInvokeInstruction c : between)
        for (CGNode target : callGraph.getPossibleTargets(caller, c.getCallSite()))
          if (region.add(target)) worklist.add(target);
      while (!worklist.isEmpty())
        for (var it = callGraph.getSuccNodes(worklist.remove()); it.hasNext(); ) {
          CGNode next = it.next();
          if (region.add(next)) worklist.add(next);
        }
      return region;
    }
    return null;
  }

  /**
   * The calls on the normal paths from {@code call} to {@code restore}, or {@code null} when some
   * normal path from {@code call} reaches the exit without passing {@code restore}.
   */
  private static List<SSAAbstractInvokeInstruction> callsBetween(
      SSACFG cfg, IR ir, SSAInstruction call, SSAInstruction restore) {
    ISSABasicBlock start = cfg.getBlockForInstruction(call.iIndex());
    ISSABasicBlock end = cfg.getBlockForInstruction(restore.iIndex());
    List<SSAAbstractInvokeInstruction> ret = new ArrayList<>();
    if (start.equals(end)) {
      // Both in one block: the restore follows the call there, or the block's own path is the one
      // that skips it.
      if (restore.iIndex() < call.iIndex()) return null;
      collectCalls(start, call.iIndex(), restore.iIndex(), ret);
      return ret;
    }
    collectCalls(start, call.iIndex(), Integer.MAX_VALUE, ret);
    // The start block is not marked seen, so a loop back into it collects its calls before `call`
    // too, which then run under the value set.
    Set<ISSABasicBlock> seen = HashSetFactory.make();
    Deque<ISSABasicBlock> worklist = new ArrayDeque<>();
    worklist.add(start);
    while (!worklist.isEmpty()) {
      ISSABasicBlock block = worklist.remove();
      for (ISSABasicBlock next : cfg.getNormalSuccessors(block)) {
        if (next.equals(end)) continue;
        if (next.isExitBlock()) return null;
        if (seen.add(next)) {
          collectCalls(next, -1, Integer.MAX_VALUE, ret);
          worklist.add(next);
        }
      }
    }
    collectCalls(end, -1, restore.iIndex(), ret);
    return ret;
  }

  /** Adds the calls of {@code block} strictly between the two instruction indices. */
  private static void collectCalls(
      ISSABasicBlock block, int after, int before, List<SSAAbstractInvokeInstruction> ret) {
    for (SSAInstruction instruction : block)
      if (instruction instanceof SSAAbstractInvokeInstruction invoke
          && instruction.iIndex() > after
          && instruction.iIndex() < before) ret.add(invoke);
  }
}
