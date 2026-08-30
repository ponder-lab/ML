package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Modeling of the function-style {@code numpy.array(x, dtype)} call. Preserves the shape of the
 * first positional argument ({@code x}) and applies the second positional argument as the output
 * dtype, mirroring {@link AstypeOperation}'s shape-preserving / dtype-changing semantics for the
 * method-style {@code x.astype(dtype)} counterpart. When no explicit {@code dtype} argument is
 * given, the dtype is inferred from the contents of {@code x} using numpy's promotion rules (<a
 * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>).
 */
public class NpArray extends TensorGenerator {
  private static final Logger LOGGER = Logger.getLogger(NpArray.class.getName());

  /**
   * The Jython runtime class representing a Python complex literal. Matched by name to avoid a
   * compile-time dependency on the Jython runtime.
   */
  private static final String PYCOMPLEX_CLASS_NAME = "org.python.core.PyComplex";

  /** The type-name shape of the front-end's synthetic comprehension functions. */
  private static final Pattern COMPREHENSION_TYPE_NAME = Pattern.compile(".*/comprehension\\d+");

  public NpArray(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public NpArray(CGNode node) {
    super(node);
  }

  /**
   * Declares a dtype feed from the source argument ({@code x}) (<a
   * href="https://github.com/wala/ML/issues/772">wala/ML#772</a>): {@code numpy.array} preserves an
   * array-like input's element dtype, so when this generator's own resolution floors at {@code
   * UNKNOWN} (an opaque, content-dependent {@code x}), dtype evidence that lives only in {@code
   * TensorTypeAnalysis} dataflow state at the argument (e.g. a type-annotation seed, <a
   * href="https://github.com/wala/ML/issues/370">wala/ML#370</a>) still reaches the result. The
   * operand key pairs {@link #getArgumentValueNumber(int)}'s value number with {@link #getNode()}'s
   * own frame, the same pairing this generator's shape and dtype reads use, so it is consistent for
   * both anchor shapes; the caller-argument edges in the assignment graph deliver the operand's
   * state there.
   *
   * <p>The feed models only the preservation case: an explicit {@code dtype} argument overrides the
   * source's dtype at runtime, so when a caller passes one that the dtype resolution could not map,
   * borrowing the operand's dtype would fabricate evidence, and no feed is declared (<a
   * href="https://github.com/wala/ML/issues/774">wala/ML#774</a>; the result stays soundly {@code
   * UNKNOWN}). The passing is detected syntactically at the caller invokes, which is decisive
   * regardless of whether the argument's own points-to evidence survives.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The {@code DTYPE_ONLY} feed from {@code x}, or {@code null} for an unlocatable argument
   *     or an explicit-but-unresolvable {@code dtype} argument.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    int sourceVn = getArgumentValueNumber(0);
    if (sourceVn <= 0) return null;
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    int dtypeVn = getArgumentValueNumber(1);
    if (dtypeVn > 0 && getDTypes(builder, dtypeVn).isEmpty() && this.isDtypeArgumentPassed(builder))
      return null;
    PointerKey argument = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), sourceVn);
    List<PointerKey> operands = new ArrayList<>();
    operands.add(argument);
    // The argument is often a container of the actual values (an append-built or literal batch
    // list), whose own variable carries no dataflow state: the element state lives in the
    // container's field keys. Add the append-contents key and the cataloged subscript keys of
    // each list/tuple allocation so element evidence feeds too.
    addContainerElementOperands(builder, argument, operands);
    return new TypeFeed(TypeFeedKind.DTYPE_ONLY, operands);
  }

  /**
   * Determines whether any caller invoke supplies a {@code dtype} argument, positionally (a second
   * positional argument beyond {@code x}) or by keyword (wala/ML#774).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the callers.
   * @return {@code true} iff some caller passes a {@code dtype} argument.
   */
  private boolean isDtypeArgumentPassed(PropagationCallGraphBuilder builder) {
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction invoke = (PythonInvokeInstruction) callerInvoke.snd;
      if (invoke.getNumberOfPositionalParameters() >= 3 || invoke.getKeywords().contains("dtype"))
        return true;
    }
    return false;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The default-mode legacy view of the record result: a partial contributes its resolvable
    // members (the wala/ML#716 contract); ⊤ and ⊥ keep their legacy encodings.
    ShapeResult result = this.getDefaultShapeResult(builder);
    return result.isPartial() ? result.members() : result.toLegacy();
  }

  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    int sourceVn = getArgumentValueNumber(0);
    LOGGER.fine(
        () -> "NpArray.getDefaultShapes: source=" + describe(source) + ", sourceVn=" + sourceVn);
    if (sourceVn > 0) {
      try {
        Set<List<Dimension<?>>> shapes = getShapes(builder, getNode(), sourceVn);
        LOGGER.fine(
            () -> "NpArray.getDefaultShapes: shapes from sourceVn=" + sourceVn + " -> " + shapes);
        if (shapes != null && !shapes.isEmpty()) {
          return ShapeResult.of(shapes);
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "NpArray.getDefaultShapes: source shape lookup failed for sourceVn=" + sourceVn,
            e);
      }
    }

    ShapeResult viaComprehension = this.getComprehensionWindowShapeResult(builder);
    if (viaComprehension != null) return viaComprehension;

    return ShapeResult.unknown();
  }

  /**
   * The windowed-batcher arm (wala/ML#851): {@code np.array} over a list built by a comprehension
   * of fixed-length windows, the hand-rolled numpy batcher shape ({@code np.array([get(f, length)
   * for f in random.sample(files, k=batch_size)])}). The content list's element values are opaque
   * (loaded data sliced to a window), so the value walks resolve nothing, yet both leading extents
   * are statically determined by the program's own text: the leading axis is the comprehension's
   * arity, which is {@code random.sample}'s {@code k}, and the second axis is each window's length,
   * recovered from the element producer's slice contract ({@code data[start : start + K]} is {@code
   * K} long).
   *
   * <p>The result is a partial ({@link ShapeResult}): the two-axis member is the resolvable subset,
   * and the unknown remainder stays set because nothing proves the element's own rank (a loaded
   * element could itself be nested), the pad branch of a guarded window is not folded, and the
   * slice contract reads the unclamped window (the subject's {@code max_length <= len(data)} guard
   * is what makes it exact at runtime, and this walk does not evaluate guards). An axis whose chase
   * declines degrades to {@link UnresolvedDim}; when neither axis resolves, the arm yields nothing
   * and the legacy ⊤ stands.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The partial result, or {@code null} when this is not a resolvable comprehension batch.
   */
  private ShapeResult getComprehensionWindowShapeResult(PropagationCallGraphBuilder builder) {
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction) || caller.getDU() == null)
        continue;
      PythonInvokeInstruction arrayCall = (PythonInvokeInstruction) callerInvoke.snd;
      if (arrayCall.getNumberOfPositionalParameters() < 2) continue;

      SSAInstruction contentDef = caller.getDU().getDef(arrayCall.getUse(1));
      if (!(contentDef instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction compInvoke = (PythonInvokeInstruction) contentDef;
      // The comprehension call passes the fresh list and the iterable to the synthetic
      // comprehension function; its function value is a fresh allocation of the comprehension's
      // own code type.
      if (compInvoke.getNumberOfPositionalParameters() != 3) continue;
      SSAInstruction compFnDef = caller.getDU().getDef(compInvoke.getUse(0));
      if (!(compFnDef instanceof SSANewInstruction)) continue;
      String compTypeName = ((SSANewInstruction) compFnDef).getConcreteType().getName().toString();
      if (!COMPREHENSION_TYPE_NAME.matcher(compTypeName).matches()) continue;

      Dimension<?> leading = comprehensionArity(builder, caller, compInvoke.getUse(2));
      Dimension<?> window = comprehensionElementExtent(builder, caller, compInvoke);
      if (leading instanceof UnresolvedDim && window instanceof UnresolvedDim) continue;

      LOGGER.fine(
          () ->
              "Comprehension window batch resolved: arity=" + leading + ", window=" + window + ".");
      Set<List<Dimension<?>>> members = new HashSet<>();
      members.add(List.of(leading, window));
      return new ShapeResult(members, true);
    }
    return null;
  }

  /**
   * Resolves the comprehension's arity: a comprehension over {@code random.sample(..., k=N)} has
   * exactly {@code N} elements when {@code N} resolves ({@code k} by keyword or the second
   * positional argument).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node holding the comprehension call.
   * @param iterableVn The comprehension's iterable value number.
   * @return The arity as a {@link NumericDim}, or {@link UnresolvedDim} when the chase declines.
   */
  private static Dimension<?> comprehensionArity(
      PropagationCallGraphBuilder builder, CGNode node, int iterableVn) {
    SSAInstruction def = node.getDU().getDef(iterableVn);
    if (def instanceof PythonInvokeInstruction) {
      PythonInvokeInstruction call = (PythonInvokeInstruction) def;
      SSAInstruction calleeDef = node.getDU().getDef(call.getUse(0));
      SymbolTable st = node.getIR().getSymbolTable();
      if (calleeDef instanceof PythonPropertyRead) {
        PythonPropertyRead read = (PythonPropertyRead) calleeDef;
        if (st.isStringConstant(read.getMemberRef())
            && "sample".equals(st.getStringValue(read.getMemberRef()))
            && readsLexicalNamed(node, read.getObjectRef(), "random")) {
          int kVn =
              call.getKeywords().contains("k")
                  ? call.getUse("k")
                  : call.getNumberOfPositionalParameters() >= 3 ? call.getUse(2) : -1;
          if (kVn > 0) {
            Integer k =
                resolveIntFlowSensitively(
                    builder, node, kVn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
            if (k != null && k > 0) return new NumericDim(k);
          }
        }
      }
    }
    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the window length of the comprehension's elements by chasing the element expression
   * into its producers: through the synthetic mapping wrapper to the comprehension body, into the
   * called producer's returns, across return-site phis, down to a slice whose bounds fold to a
   * length. Opaque leaves (an unmodeled load, the pad loop) are tolerated &mdash; the caller's
   * unknown remainder covers them &mdash; but resolved leaves must agree.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param caller The node holding the comprehension call.
   * @param compInvoke The comprehension invoke.
   * @return The window length as a {@link NumericDim}, or {@link UnresolvedDim}.
   */
  private static Dimension<?> comprehensionElementExtent(
      PropagationCallGraphBuilder builder, CGNode caller, PythonInvokeInstruction compInvoke) {
    Integer agreed = null;
    Set<Pair<CGNode, Integer>> visited = new HashSet<>();
    for (CGNode wrapper :
        builder.getCallGraph().getPossibleTargets(caller, compInvoke.getCallSite())) {
      // The wrapper is the synthetic per-element mapping trampoline; its call sites reach the
      // comprehension body.
      for (Iterator<CallSiteReference> sites = wrapper.iterateCallSites(); sites.hasNext(); ) {
        CallSiteReference site = sites.next();
        for (CGNode body : builder.getCallGraph().getPossibleTargets(wrapper, site)) {
          Integer extent =
              returnedLeadingExtent(builder, body, visited, FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
          if (extent == null) continue;
          if (agreed != null && !agreed.equals(extent)) return UnresolvedDim.INSTANCE;
          agreed = extent;
        }
      }
    }
    return agreed == null ? UnresolvedDim.INSTANCE : new NumericDim(agreed);
  }

  /**
   * Resolves the agreeing leading extent of a node's returned values; opaque returns are tolerated,
   * resolved ones must agree.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node whose returns are chased.
   * @param visited The {@code (node, vn)} pairs already on this chase, breaking cycles.
   * @param depth The remaining recursion budget.
   * @return The agreeing extent, or {@code null}.
   */
  private static Integer returnedLeadingExtent(
      PropagationCallGraphBuilder builder,
      CGNode node,
      Set<Pair<CGNode, Integer>> visited,
      int depth) {
    if (depth <= 0 || node.getIR() == null || node.getDU() == null) return null;
    Integer agreed = null;
    for (SSAInstruction inst : node.getIR().getInstructions()) {
      if (!(inst instanceof SSAReturnInstruction)) continue;
      int resultVn = ((SSAReturnInstruction) inst).getResult();
      if (resultVn <= 0) continue;
      Integer extent = leadingExtentOfValue(builder, node, resultVn, visited, depth);
      if (extent == null) continue;
      if (agreed != null && !agreed.equals(extent)) return null;
      agreed = extent;
    }
    return agreed;
  }

  /**
   * Resolves one value's leading extent: a phi takes its operands' agreeing extents (opaque
   * operands tolerated), a call to the {@code slice} builtin folds its bounds, and any other call
   * chases the callees' returns.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node whose value is chased.
   * @param vn The value number to chase.
   * @param visited The {@code (node, vn)} pairs already on this chase, breaking cycles.
   * @param depth The remaining recursion budget.
   * @return The extent, or {@code null} for an opaque value.
   */
  private static Integer leadingExtentOfValue(
      PropagationCallGraphBuilder builder,
      CGNode node,
      int vn,
      Set<Pair<CGNode, Integer>> visited,
      int depth) {
    if (vn <= 0 || depth <= 0 || !visited.add(Pair.make(node, vn))) return null;
    SSAInstruction def = node.getDU().getDef(vn);

    if (def instanceof SSAPhiInstruction) {
      Integer agreed = null;
      for (int i = 0; i < def.getNumberOfUses(); i++) {
        Integer operand = leadingExtentOfValue(builder, node, def.getUse(i), visited, depth - 1);
        if (operand == null) continue;
        if (agreed != null && !agreed.equals(operand)) return null;
        agreed = operand;
      }
      return agreed;
    }

    if (def instanceof PythonInvokeInstruction) {
      PythonInvokeInstruction call = (PythonInvokeInstruction) def;
      if (readsLexicalNamed(node, call.getUse(0), "slice")) return sliceExtent(builder, node, call);

      Integer agreed = null;
      for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, call.getCallSite())) {
        Integer viaCallee = returnedLeadingExtent(builder, callee, visited, depth - 1);
        if (viaCallee == null) continue;
        if (agreed != null && !agreed.equals(viaCallee)) return null;
        agreed = viaCallee;
      }
      return agreed;
    }

    return null;
  }

  /**
   * Folds a {@code slice(obj, start, stop, step)} builtin call's extent: with a unit step, {@code
   * obj[start : start + K]} is {@code K} long by operand identity, and literal bounds subtract. The
   * result is the unclamped window; see the arm's contract for why that is the emitted member.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node holding the slice call.
   * @param call The slice call.
   * @return The window length, or {@code null} when the bounds do not fold.
   */
  private static Integer sliceExtent(
      PropagationCallGraphBuilder builder, CGNode node, PythonInvokeInstruction call) {
    if (call.getNumberOfPositionalParameters() < 4) return null;
    SymbolTable st = node.getIR().getSymbolTable();
    if (call.getNumberOfPositionalParameters() >= 5) {
      int stepVn = call.getUse(4);
      boolean unitStep =
          st.isNullConstant(stepVn)
              || (st.isNumberConstant(stepVn)
                  && ((Number) st.getConstantValue(stepVn)).longValue() == 1L);
      if (!unitStep) return null;
    }

    int startVn = call.getUse(2);
    int stopVn = call.getUse(3);
    SSAInstruction stopDef = node.getDU().getDef(stopVn);
    if (stopDef instanceof SSABinaryOpInstruction
        && ((SSABinaryOpInstruction) stopDef).getOperator() == IBinaryOpInstruction.Operator.ADD) {
      int a = stopDef.getUse(0);
      int b = stopDef.getUse(1);
      int lengthVn = a == startVn ? b : b == startVn ? a : -1;
      if (lengthVn > 0) {
        // The int chase carries its own cycle guard and needs its own full budget: threading this
        // walk's depleted depth in starves the caller-agreement hops (parameter, trampoline,
        // lexical-definer) the length routinely crosses.
        Integer length =
            resolveIntFlowSensitively(
                builder, node, lengthVn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
        if (length != null && length > 0) return length;
        return null;
      }
    }

    Integer start =
        resolveIntFlowSensitively(
            builder, node, startVn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
    Integer stop =
        resolveIntFlowSensitively(
            builder, node, stopVn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
    if (start != null && stop != null && stop > start && start >= 0) return stop - start;
    return null;
  }

  /**
   * Decides whether a value is a lexical read of the given name (the {@code enumerate}-match
   * idiom).
   *
   * @param node The node whose value is inspected.
   * @param vn The value number to inspect.
   * @param name The variable name to match.
   * @return {@code true} iff the value reads that lexical name.
   */
  private static boolean readsLexicalNamed(CGNode node, int vn, String name) {
    SSAInstruction def = node.getDU().getDef(vn);
    return def instanceof AstLexicalRead
        && ((AstLexicalRead) def).getAccessCount() > 0
        && name.equals(((AstLexicalRead) def).getAccess(0).getName().fst);
  }

  @Override
  @SuppressWarnings("unchecked")
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // The explicit `dtype` argument is a token (`np.int32`, the builtin `int`), not a value whose
    // dtype the value walk could compute (that walk deliberately ignores `DType` allocations), so
    // it resolves through the dtype-argument token resolver, with the argument located by the
    // multi-strategy resolver: the callee-frame parameter's points-to set is empty on a
    // return-value anchoring, and the caller walk recovers the keyword (wala/ML#775).
    OrdinalSet<InstanceKey> dtypePTS = getArgumentPointsToSet(builder, 1, "dtype");
    LOGGER.fine(() -> "Resolved dtype argument points-to set: " + describe(dtypePTS) + ".");
    if (dtypePTS != null && !dtypePTS.isEmpty()) {
      Set<DType> dTypes = getDTypesFromDTypeArgument(builder, dtypePTS);
      if (!dTypes.isEmpty()) {
        return dTypes;
      }
    }

    // No explicit `dtype` argument: infer from the contents of `x` (arg 0) using numpy's promotion
    // rules. numpy promotes Python `int` to `int64` and `float` to `float64` (not the `int32` /
    // `float32` TF-literal convention), so a numpy-specific walk is needed. wala/ML#626.
    int sourceVn = getArgumentValueNumber(0);
    if (sourceVn > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(getNode(), sourceVn);
      OrdinalSet<InstanceKey> sourcePTS = pa.getPointsToSet(pk);
      Set<DType> inferred = numpyPromotedDTypes(builder, sourcePTS);
      if (!inferred.isEmpty() && !(inferred.size() == 1 && inferred.contains(DType.UNKNOWN))) {
        LOGGER.fine(() -> "Inferred " + inferred + " from the content argument's leaves.");
        return inferred;
      }

      // The content's leaves resolve to nothing or only the ⊤ floor (e.g. a rebased element whose
      // producers are only reachable syntactically, the reader-chain merge frames): fall back to
      // the SSA-chain walk, whose lexical, phi, tuple-peel, and parameter arms recover the
      // producing array's dtype interprocedurally. `np.array` preserves an array-typed content's
      // dtype, so the recovered dtype is the result's. The walk is routed through the engine so
      // the transfer is schedule-visible: a seed-time evaluation that runs before the upstream
      // generators exist re-evaluates when the values it read grow, instead of memoizing the
      // unknown forever (the wala/ML#753 class; `readElementShapes` is the precedent).
      // wala/ML#796.
      WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
      Set<DType> viaChain;
      if (engine == null) viaChain = getDTypesOrSSAChain(builder, getNode(), sourceVn);
      else {
        Object key = Pair.make("nparray-content-chain", pk);
        java.util.function.Supplier<Object> transfer =
            () -> getDTypesOrSSAChain(builder, getNode(), sourceVn);
        viaChain =
            (Set<DType>)
                (engine.isEvaluating()
                    ? engine.read(key, transfer, false)
                    : engine.demand(key, transfer, false));
      }
      // Only a SINGLE resolved dtype is content evidence. `np.array` promotes heterogeneous
      // content to one dtype rather than producing a union, so a multi-dtype chain result cannot
      // describe the content's element type: in the corpus it means the walk reached a container
      // of mixed constants (an object catalog or a keyword map, one dtype per Java constant kind)
      // and widened edge-weight parameters to a `STRING`-bearing union (wala/ML#806).
      if (viaChain != null && viaChain.size() == 1 && !viaChain.contains(DType.UNKNOWN)) {
        LOGGER.fine(() -> "Recovered " + viaChain + " via the content argument's SSA chain.");
        return viaChain;
      }
      if (viaChain != null && viaChain.size() > 1)
        LOGGER.fine(
            () ->
                "Discarding the content argument's multi-dtype chain result as non-evidence: "
                    + viaChain
                    + ".");
    }

    return Set.of(DType.UNKNOWN);
  }

  /**
   * Infers the numpy-promoted dtype of a literal {@code x} argument from its leaf scalar values.
   * The widest leaf kind wins, following numpy's promotion order (a string anywhere yields a string
   * array; otherwise complex {@literal >} float {@literal >} int {@literal >} bool). Returns the
   * empty set when no leaf type is recoverable, and {@code {UNKNOWN}} when {@code x} (or a nested
   * element) is not a literal the walk can promote (e.g. an existing array or tensor, whose dtype
   * numpy would preserve rather than promote) &mdash; ⊤ is the sound floor there. wala/ML#626.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param sourcePTS The points-to set of the {@code x} argument.
   * @return The promoted dtype, an {@code {UNKNOWN}} floor, or the empty set.
   */
  private Set<DType> numpyPromotedDTypes(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> sourcePTS) {
    if (sourcePTS == null || sourcePTS.isEmpty()) return Set.of();
    LOGGER.fine(() -> "numpyPromotedDTypes: walking " + describe(sourcePTS) + ".");

    EnumSet<DType> leaves = EnumSet.noneOf(DType.class);
    if (!collectNumpyLeaves(builder, sourcePTS, leaves, new HashSet<>()))
      return Set.of(DType.UNKNOWN);

    // A single leaf kind needs no promotion: this covers dtype preservation from an existing
    // array/tensor element (wala/ML#796), whose non-literal dtype (e.g. int32) the ladder below
    // does not rank.
    if (leaves.size() == 1) return EnumSet.copyOf(leaves);

    // Promotion is modeled only over the literal leaf kinds; a mix involving a preserved
    // non-literal dtype floors to ⊤ rather than mis-ranking.
    if (!EnumSet.of(DType.STRING, DType.COMPLEX128, DType.FLOAT64, DType.INT64, DType.BOOL)
        .containsAll(leaves)) return Set.of(DType.UNKNOWN);

    // Promotion order: a string array subsumes everything; otherwise widen numerically.
    if (leaves.contains(DType.STRING)) return Set.of(DType.STRING);
    if (leaves.contains(DType.COMPLEX128)) return Set.of(DType.COMPLEX128);
    if (leaves.contains(DType.FLOAT64)) return Set.of(DType.FLOAT64);
    if (leaves.contains(DType.INT64)) return Set.of(DType.INT64);
    if (leaves.contains(DType.BOOL)) return Set.of(DType.BOOL);
    return Set.of();
  }

  /**
   * Recursively collects the numpy base dtypes of the leaf scalars reachable from {@code pts},
   * descending through nested {@code list}/{@code tuple} allocations.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pts The points-to set to walk.
   * @param leaves Accumulator for the leaf numpy base dtypes ({@code BOOL}, {@code INT64}, {@code
   *     FLOAT64}, {@code COMPLEX128}, {@code STRING}).
   * @param visited The instance keys already descended through, guarding against points-to cycles
   *     among container allocations (wala/ML#796).
   * @return {@code true} if every element was a literal scalar, a nested list/tuple of literals, or
   *     an existing array/tensor whose producer resolves a definite dtype (preservation,
   *     wala/ML#796); {@code false} otherwise, in which case the caller floors to ⊤.
   */
  private boolean collectNumpyLeaves(
      PropagationCallGraphBuilder builder,
      OrdinalSet<InstanceKey> pts,
      EnumSet<DType> leaves,
      Set<InstanceKey> visited) {
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey) && !visited.add(ik)) continue;
      if (ik instanceof ConstantKey) {
        Object value = ((ConstantKey<?>) ik).getValue();
        if (value == null) {
          // A null constant next to real members is analysis substrate (loop-carried phi
          // defaults, the generator iteration protocol), not element evidence; flooring on it
          // would erase the sibling members' dtypes (wala/ML#796). A value that is only null
          // collects nothing and degrades to ⊤ via the empty-leaves path.
          LOGGER.fine(() -> "collectNumpyLeaves: skipping null constant.");
          continue;
        }
        if (value instanceof Float || value instanceof Double) leaves.add(DType.FLOAT64);
        else if (value instanceof Boolean) leaves.add(DType.BOOL);
        else if (value instanceof Integer || value instanceof Long) leaves.add(DType.INT64);
        else if (value instanceof String) leaves.add(DType.STRING);
        else if (PYCOMPLEX_CLASS_NAME.equals(value.getClass().getName()))
          // A Python complex literal, which the Jython front-end represents as a `PyComplex`.
          leaves.add(DType.COMPLEX128);
        else {
          LOGGER.fine(() -> "collectNumpyLeaves: unrecognized scalar " + value + "; flooring.");
          return false; // Unrecognized scalar.
        }
      } else {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null) {
          LOGGER.fine(() -> "collectNumpyLeaves: no allocation site for " + ik + "; flooring.");
          return false;
        }

        TypeReference reference = asin.concreteType().getReference();
        if (!reference.equals(list) && !reference.equals(tuple)) {
          // An existing array/tensor: numpy preserves its dtype rather than promoting. Delegate
          // to the producer for the preserved dtype; floor only when the producer cannot resolve
          // one (wala/ML#796).
          Set<DType> preserved = this.getDTypesFromTensor(builder, asin);
          LOGGER.fine(
              () ->
                  "collectNumpyLeaves: producer of "
                      + asin.concreteType().getReference().getName()
                      + " preserved "
                      + preserved
                      + ".");
          if (preserved == null || preserved.isEmpty() || preserved.contains(DType.UNKNOWN))
            return false;
          leaves.addAll(preserved);
          continue;
        }

        OrdinalSet<InstanceKey> catalogPTS =
            pa.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : catalogPTS) {
          Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
          if (fieldIndex == null) continue; // Skip non-integer attribute keys. wala/ML#603.

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f == null) continue;

          OrdinalSet<InstanceKey> fieldPTS =
              pa.getPointsToSet(builder.getPointerKeyForInstanceField(asin, f));
          if (!collectNumpyLeaves(builder, fieldPTS, leaves, visited)) return false;
        }
      }
    }

    return true;
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
    return 1;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }

  /**
   * Returns the producing library of the modeled value: an }np.array(...)` call, so the value is an
   * ndarray (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
