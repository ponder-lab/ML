package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
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
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The {@code DTYPE_ONLY} feed from {@code x}, or {@code null} for an unlocatable
   *     argument.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    int sourceVn = getArgumentValueNumber(0);
    if (sourceVn <= 0) return null;
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    PointerKey argument = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), sourceVn);
    List<PointerKey> operands = new ArrayList<>();
    operands.add(argument);
    // The argument is often a container of the actual values (an append-built or literal batch
    // list), whose own variable carries no dataflow state: the element state lives in the
    // container's field keys. Add the append-contents key and the cataloged subscript keys of
    // each list/tuple allocation so element evidence feeds too.
    for (InstanceKey ik : pa.getPointsToSet(argument)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      TypeReference reference = asin.concreteType().getReference();
      if (!reference.equals(list) && !reference.equals(tuple)) continue;
      FieldReference contents =
          FieldReference.findOrCreate(
              Root,
              findOrCreateAsciiAtom(
                  PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD),
              Root);
      IField contentsField = builder.getClassHierarchy().resolveField(contents);
      if (contentsField != null)
        operands.add(builder.getPointerKeyForInstanceField(asin, contentsField));
      OrdinalSet<InstanceKey> catalogPTS =
          pa.getPointsToSet(
              ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                  .getPointerKeyForObjectCatalog(asin));
      for (InstanceKey catalogIK : catalogPTS) {
        if (!(catalogIK instanceof ConstantKey)) continue;
        Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
        if (fieldIndex == null) continue;
        FieldReference subscript =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
        IField subscriptField = builder.getClassHierarchy().resolveField(subscript);
        if (subscriptField != null)
          operands.add(builder.getPointerKeyForInstanceField(asin, subscriptField));
      }
    }
    return new TypeFeed(TypeFeedKind.DTYPE_ONLY, operands);
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
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int dtypeVn = getArgumentValueNumber(1);
    if (dtypeVn > 0) {
      Set<DType> dTypes = getDTypes(builder, dtypeVn);
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
      if (!inferred.isEmpty()) {
        LOGGER.fine(() -> "NpArray.getDefaultDTypes: inferred " + inferred + " from sourceVn.");
        return inferred;
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

    EnumSet<DType> leaves = EnumSet.noneOf(DType.class);
    if (!collectNumpyLeaves(builder, sourcePTS, leaves)) return Set.of(DType.UNKNOWN);

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
   * @return {@code true} if every element was a literal scalar or a nested list/tuple of literals;
   *     {@code false} if an element is not a promotable literal (e.g. an existing array/tensor), in
   *     which case the caller floors to ⊤.
   */
  private boolean collectNumpyLeaves(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> pts, EnumSet<DType> leaves) {
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

    for (InstanceKey ik : pts) {
      if (ik instanceof ConstantKey) {
        Object value = ((ConstantKey<?>) ik).getValue();
        if (value instanceof Float || value instanceof Double) leaves.add(DType.FLOAT64);
        else if (value instanceof Boolean) leaves.add(DType.BOOL);
        else if (value instanceof Integer || value instanceof Long) leaves.add(DType.INT64);
        else if (value instanceof String) leaves.add(DType.STRING);
        else if (value != null && "org.python.core.PyComplex".equals(value.getClass().getName()))
          // A Python complex literal, which the Jython front-end represents as a `PyComplex{@code .
          // Matched by class name to avoid a compile-time dependency on the Jython runtime.
          leaves.add(DType.COMPLEX128);
        else return false; // Unrecognized scalar.
      } else {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null) return false;

        TypeReference reference = asin.concreteType().getReference();
        if (!reference.equals(list) && !reference.equals(tuple))
          // An existing array/tensor (or other non-literal): numpy would preserve its dtype rather
          // than promote, which this walk does not model. Floor to ⊤.
          return false;

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
          if (!collectNumpyLeaves(builder, fieldPTS, leaves)) return false;
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
