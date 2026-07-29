package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

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
    int sourceVn = getArgumentValueNumber(0);
    LOGGER.fine(
        () -> "NpArray.getDefaultShapes: source=" + describe(source) + ", sourceVn=" + sourceVn);
    if (sourceVn > 0) {
      try {
        Set<List<Dimension<?>>> shapes = getShapes(builder, getNode(), sourceVn);
        LOGGER.fine(
            () -> "NpArray.getDefaultShapes: shapes from sourceVn=" + sourceVn + " -> " + shapes);
        if (shapes != null && !shapes.isEmpty()) {
          return shapes;
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "NpArray.getDefaultShapes: source shape lookup failed for sourceVn=" + sourceVn,
            e);
      }
    }
    return null;
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
      if (viaChain != null
          && !viaChain.isEmpty()
          && !(viaChain.size() == 1 && viaChain.contains(DType.UNKNOWN))) {
        LOGGER.fine(() -> "Recovered " + viaChain + " via the content argument's SSA chain.");
        return viaChain;
      }
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
