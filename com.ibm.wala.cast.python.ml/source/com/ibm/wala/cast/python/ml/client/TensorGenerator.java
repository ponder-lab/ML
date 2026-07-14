package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine.TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_CHOOSE_FROM_DATASETS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SAMPLE_FROM_DATASETS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.BOOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UNKNOWN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIELD_REFERENCE_TO_DTYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.PLACEHOLDER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME_FOR_KERAS_MODELS;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.KERAS_BUILD_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static com.ibm.wala.cast.python.util.Util.getReceiverValueNumber;
import static com.ibm.wala.cast.python.util.Util.sanitize;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.Collections.emptyList;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ir.ssa.CAstUnaryOp;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.CompoundDim;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.Layout;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.ReturnValueKey;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.WeakHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An abstract generator for {@link TensorType}s.
 *
 * <h2>Lattice conventions for shapes and dtypes</h2>
 *
 * Subclasses <strong>must</strong> follow these conventions so that downstream consumers can
 * distinguish "unknown tensor" (⊤) from "not a tensor" (⊥):
 *
 * <h3>Shapes — {@link #getDefaultShapes(PropagationCallGraphBuilder)}</h3>
 *
 * <ul>
 *   <li>{@code null} — ⊤, the generator produces a tensor but its shape cannot be determined.
 *   <li>empty set ({@code Collections.emptySet()}) — ⊥, the variable is provably not a tensor.
 *   <li>non-empty set — the set of concrete shapes the tensor may take.
 * </ul>
 *
 * <p>Within a single shape, use {@link TensorType.SymbolicDim}{@code ("?")} for a
 * known-rank-but-unknown-size dimension (e.g., a dynamic batch size). A {@code null} shape list
 * means even the rank is unknown.
 *
 * <h3>Dtypes — {@link #getDefaultDTypes(PropagationCallGraphBuilder)}</h3>
 *
 * <ul>
 *   <li>{@code EnumSet.of(DType.UNKNOWN)} — ⊤, the generator produces a tensor but its dtype cannot
 *       be determined. Never return a bare empty set for the "unknown" case.
 *   <li>empty set — ⊥, the variable is provably not a tensor.
 *   <li>non-empty set of concrete {@link DType}s — the set of possible dtypes.
 * </ul>
 *
 * <h3>Tensor types — {@link #getTensorTypes(PropagationCallGraphBuilder)}</h3>
 *
 * Shapes and dtypes are orthogonal. When the shape is unknown but the dtype is known, {@link
 * #getTensorTypes(PropagationCallGraphBuilder)} emits {@code TensorType} instances with {@code
 * null} dims so dtype information is preserved. {@link TensorType} is null-dims-safe; subclasses
 * consuming {@link TensorType}s must also be.
 *
 * <p>When adding a new {@link TensorGenerator} subclass, audit every final-fallback return in
 * {@code getDefaultShapes} and {@code getDefaultDTypes} against the table above — the most common
 * mistake is returning {@code Collections.emptySet()} when the intended meaning is "unknown."
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class TensorGenerator {

  protected static final int UNDEFINED_PARAMETER_POSITION = -1;

  private static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  /**
   * A shape-resolution query key (wala/ML#716): the queried value plus the resolution mode.
   * Exact-mode results poison to ⊤ whenever any points-to member fails to resolve, so they cannot
   * share engine state entries with the default partial-union results.
   *
   * @param node The {@link CGNode} whose IR defines the value.
   * @param valueNumber The queried value number.
   * @param exact Whether an unresolvable union member poisons the result to ⊤.
   */
  private record ShapeQuery(CGNode node, int valueNumber, boolean exact) {}

  /**
   * Returns the memo key identifying a generator evaluation (wala/ML#365).
   *
   * @param generator The generator whose evaluation is keyed.
   * @return The concrete class paired with the anchor identity.
   */
  private static Object evaluationKey(TensorGenerator generator) {
    Object anchorId =
        generator.getSource() != null
            ? generator.getSource().getPointerKey()
            : generator.manualNode;
    return Pair.make(generator.getClass(), anchorId);
  }

  /**
   * Evaluates the generator's shapes through the worklist engine (wala/ML#365), which memoizes,
   * records the dependency edge, and converges cycles.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param generator The generator to evaluate.
   * @return The evaluation result.
   */
  static ShapeResult memoizedShapeResult(
      PropagationCallGraphBuilder builder, TensorGenerator generator) {
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    // The engine is installed for every analysis run's resolution (wala/ML#365, Phase 3); a null
    // engine occurs only for direct generator use outside one (e.g. unit tests) and computes
    // unmemoized.
    if (engine == null) return generator.getShapeResult(builder);
    Object key = Pair.make("shape-eval", evaluationKey(generator));
    java.util.function.Supplier<Object> transfer = () -> generator.getShapeResult(builder);
    return (ShapeResult)
        (engine.isEvaluating()
            ? engine.read(key, transfer, true)
            : engine.demand(key, transfer, true));
  }

  /**
   * Dtype counterpart of {@link #memoizedShapeResult(PropagationCallGraphBuilder,
   * TensorGenerator)}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param generator The generator to evaluate.
   * @return The evaluation result; may be {@code null} (⊤) per the legacy dtype convention.
   */
  static Set<DType> memoizedDTypes(PropagationCallGraphBuilder builder, TensorGenerator generator) {
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    // See memoizedShapeResult on the null-engine (outside-an-analysis) fallback.
    if (engine == null) return generator.getDTypes(builder);
    Object key = Pair.make("dtype-eval", evaluationKey(generator));
    java.util.function.Supplier<Object> transfer = () -> generator.getDTypes(builder);
    @SuppressWarnings("unchecked")
    Set<DType> diverted =
        (Set<DType>)
            (engine.isEvaluating()
                ? engine.read(key, transfer, false)
                : engine.demand(key, transfer, false));
    return diverted;
  }

  /**
   * Memo of the {@code (class, attribute)} chase results of {@link
   * #resolveStoredAttributeDimInClass(PropagationCallGraphBuilder, IClass, String)} per builder
   * (wala/ML#712). The chase scans the whole call graph, so uncached re-runs are quadratic on large
   * analyses; the {@link Optional} payload caches the negative result too.
   */
  private static final Map<
          PropagationCallGraphBuilder, Map<Pair<IClass, String>, Optional<Dimension<?>>>>
      STORED_ATTRIBUTE_CACHE = Collections.synchronizedMap(new WeakHashMap<>());

  /**
   * Memo of the per-class contract results of {@link
   * #explicitBuildContractShapes(PropagationCallGraphBuilder, CGNode)} per builder (wala/ML#717).
   * The recognizer scans the whole call graph, so uncached re-runs are quadratic on large analyses;
   * the {@link Optional} payload caches the negative result too.
   */
  private static final Map<
          PropagationCallGraphBuilder, Map<IClass, Optional<Set<List<Dimension<?>>>>>>
      BUILD_CONTRACT_CACHE = Collections.synchronizedMap(new WeakHashMap<>());

  /**
   * Drops the shape/dtype caches for the given builder. Intended to be called at the end of an
   * analysis to release cache memory deterministically, rather than waiting for the builder to be
   * garbage-collected. Safe to call more than once.
   *
   * @param builder The builder whose cache entries should be cleared.
   * @implNote {@link #WARNED_UNREGISTERED_MANUAL_TYPES} is intentionally <em>not</em> cleared here.
   *     The dedup is at JVM scope so each unregistered type warns exactly once total &mdash;
   *     re-warning per test would still flood logs (~1500 firings on the current suite). The set is
   *     small and the goal is "make each gap audible once," not "track gaps per analysis."
   */
  public static void clearCaches(PropagationCallGraphBuilder builder) {
    STORED_ATTRIBUTE_CACHE.remove(builder);
    BUILD_CONTRACT_CACHE.remove(builder);
    QueryDependencyGraph.clear(builder);
  }

  /** The source of the tensor, represented by a points-to set variable. */
  protected PointsToSetVariable source;

  /**
   * The call graph node representing the "manual" generator. A generator is considered manual when
   * it is instantiated directly from a Call Graph Node ({@link CGNode}) rather than a points-to set
   * variable. This fallback mechanism is used when WALA's pointer analysis cannot construct a
   * trackable points-to set for the tensor's allocation site (often due to limitations dealing with
   * implicit allocations in synthetic model methods). In such cases, the generator analyzes the
   * instructions directly within this node's IR.
   */
  protected CGNode manualNode;

  /**
   * Constructs a new tensor generator based on a standard points-to set source.
   *
   * @param source The points-to set variable representing the source of the tensor.
   */
  public TensorGenerator(PointsToSetVariable source) {
    this.source = source;
  }

  /**
   * Constructs a new "manual" tensor generator based directly on a call graph node.
   *
   * @param node The call graph node representing the operation that generates the tensor. Used as a
   *     fallback when standard pointer analysis fails to provide a trackable source.
   */
  public TensorGenerator(CGNode node) {
    this.manualNode = node;
  }

  /**
   * Returns a set of possible {@link TensorType}s that this generator can produce, or {@code null}
   * if this generator is known to produce a tensor but its shape cannot be determined (unknown /
   * top). An empty set means the variable has no possible tensor type (i.e., it is not a tensor).
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A set of possible {@link TensorType}s, or {@code null} if the shape is unknown.
   */
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    ShapeResult shapes = memoizedShapeResult(builder, this);
    Set<DType> dTypes = memoizedDTypes(builder, this);

    // If we have no dtype info at all, fall back to signaling "unknown tensor" when shapes are
    // also unknown, otherwise produce an empty set (⊥, not a tensor).
    if (dTypes == null || dTypes.isEmpty()) {
      return shapes.members().isEmpty() && shapes.hasUnknown() ? null : HashSetFactory.make();
    }

    Set<TensorType> ret = HashSetFactory.make();

    final Layout layout = this.producesSparseTensor() ? Layout.SPARSE : Layout.DENSE;

    // Create a tensor type for each resolved shape and dtype combination; an unknown remainder
    // additionally emits null-dims (⊤-shape) types so the dtype information is preserved and a
    // partially resolved set keeps its concrete members alongside the unknown one (wala/ML#718).
    if (shapes.hasUnknown()) for (DType dtype : dTypes) ret.add(TensorType.of(dtype, null, layout));
    for (List<Dimension<?>> dimensionList : shapes.members())
      for (DType dtype : dTypes) ret.add(TensorType.of(dtype, dimensionList, layout));

    LOGGER.fine("Generator " + this.getClass().getSimpleName() + " produced types: " + ret);

    return ret;
  }

  /**
   * Whether this generator produces a sparse tensor (a {@code tf.sparse.SparseTensor} / {@code
   * tf.SparseTensor}), as opposed to a dense one. Sparse generators override this to {@code true};
   * {@link #getTensorTypes} then emits a sparse {@link TensorType} for every result so a consumer
   * can distinguish a sparse result from a dense one (<a
   * href="https://github.com/wala/ML/issues/588">wala/ML#588</a>).
   *
   * @return {@code true} if the produced tensor is sparse; {@code false} (the default) for dense.
   */
  protected boolean producesSparseTensor() {
    return false;
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * <p>An unrecognized shape form (a runtime tensor such as the result of {@code tf.shape(y)}, an
   * opaque builder value, or anything that isn't a list/tuple, {@code tf.constant}, {@code
   * TensorSpec}, or {@code RaggedTensorSpec}) degrades to {@code null} (lattice ⊤, "tensor of
   * unknown shape") rather than throwing, so the analysis doesn't abort on otherwise-valid programs
   * (<a href="https://github.com/wala/ML/issues/471">wala/ML#471</a>). Callers that depend on the
   * {@code shape} argument (e.g. {@link BroadcastTo}, {@link Reshape}) propagate that ⊤ rather than
   * unsoundly falling back to input-shape inference; the {@code FINE}-level log records the
   * unrecognized form for diagnosing modeling gaps.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the shape argument. FIXME: Why not take a value number?
   * @return A set of possible shapes, or {@code null} (lattice ⊤) when the shape cannot be resolved
   *     statically. This covers sub-parse failures during recursive descent (an empty {@code
   *     tf.constant} value PTS after the call-site walk, an empty {@link
   *     com.ibm.wala.cast.python.ml.types.TensorFlowTypes#TENSOR_SPEC} or {@link
   *     com.ibm.wala.cast.python.ml.types.TensorFlowTypes#RAGGED_TENSOR_SPEC} shape field, or a
   *     recursive call returning {@code null}) and unrecognized forms — a runtime tensor (e.g. the
   *     result of {@code tf.shape(y)}), an opaque builder object, or anything that isn't a
   *     list/tuple, {@code tf.constant}, {@code TensorSpec}, or {@code RaggedTensorSpec}. Returning
   *     ⊤ for unrecognized forms (rather than throwing) keeps the analysis from aborting on
   *     otherwise-valid programs (wala/ML#471). Non-allocation {@code InstanceKey}s in the PTS
   *     (e.g. a {@code ConstantKey} for a scalar Python int passed as the shape) are silently
   *     skipped; if every key in the PTS is non-allocation, the method returns the empty set rather
   *     than {@code null}.
   *     <p>Non-integer object-catalog keys (attribute-name fields like {@code "read_data"} that a
   *     virtual-dispatch read can leave on a list/tuple allocation) are skipped, and the leading
   *     dimension counts only the integer-indexed entries; see wala/ML#603.
   * @throws IllegalArgumentException when the {@code pointsToSet} parameter is itself empty or
   *     {@code null}. That's a caller-contract violation (callers should check for empty PTS before
   *     invoking this helper); distinct from "shape was a runtime tensor."
   */
  protected Set<List<Dimension<?>>> getShapesFromShapeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    if (pointsToSet == null || !pointsToSet.iterator().hasNext())
      throw new IllegalArgumentException(
          "Empty points-to set for shape argument in source: " + describe(this.getSource()) + ".");

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      if (asin == null) continue;
      TypeReference reference = asin.concreteType().getReference();

      if (reference.equals(dict)) {
        // A dict-structured shape specification, e.g. `padded_shapes={'h_r': [None], 't':
        // [None]}`. The values (keyed by arbitrary string names) are the per-leaf shape specs;
        // recurse into each value and union the results, mirroring the dict-structured dtype
        // handling (wala/ML#615). See wala/ML#673.
        OrdinalSet<InstanceKey> dictCatalogPts =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : dictCatalogPts) {
          if (!(catalogIK instanceof ConstantKey)) continue;
          Object keyValue = ((ConstantKey<?>) catalogIK).getValue();
          // Dict keys are strings; the value is stored as an instance field named by the key.
          if (!(keyValue instanceof String)) continue;

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom((String) keyValue), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f != null) {
            PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
            OrdinalSet<InstanceKey> fieldPts = pointerAnalysis.getPointsToSet(pk);
            if (fieldPts == null || fieldPts.isEmpty()) continue;
            Set<List<Dimension<?>>> sub = this.getShapesFromShapeArgument(builder, fieldPts);
            if (sub == null) return null;
            ret.addAll(sub);
          }
        }
        continue;
      }

      if (reference.equals(list) || reference.equals(tuple)) {
        // We have a list of integers that represent the shape.
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        // We expect the object catalog to contain a list of integers. Each element in the array
        // correspondences to the set of possible dimensions for that index.
        int elementCount = integerCatalogSize(objectCatalogPointsToSet);
        if (elementCount == 0) {
          ret.add(Collections.emptyList());
          continue;
        }
        @SuppressWarnings({"unchecked", "rawtypes"})
        Set<Dimension<?>>[] possibleDimensions = new Set[elementCount];

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Integer fieldIndex = getFieldIndex(constantKey);
          // Skip non-integer attribute keys (e.g. method-name fields); they aren't elements.
          // See wala/ML#603.
          if (fieldIndex == null) continue;

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          LOGGER.fine("Found field: " + f);

          // We can now get the pointer key for the instance field.
          PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
          LOGGER.fine(
              "Found pointer key for instance field: "
                  + describe(pointerKeyForInstanceField)
                  + ".");

          // Get the points-to set for the instance field.
          OrdinalSet<InstanceKey> instanceFieldPointsToSet =
              pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
          LOGGER.fine(
              "Points-to set for instance field: " + describe(instanceFieldPointsToSet) + ".");

          // If the instance field points to a constant, we can use it as the shape.
          // TODO: Is it possible to also do it for (simple) expressions?
          Set<Dimension<?>> tensorDimensions = HashSetFactory.make();

          for (InstanceKey instanceFieldIK : instanceFieldPointsToSet) {
            if (instanceFieldIK instanceof ConstantKey) {
              // We have a constant key.
              ConstantKey<?> instanceFieldConstant = (ConstantKey<?>) instanceFieldIK;
              Object instanceFieldValue = instanceFieldConstant.getValue();

              // We have a shape value.
              Number shapeValue = (Number) instanceFieldValue;
              LOGGER.fine(
                  "Found shape value: "
                      + shapeValue
                      + " for "
                      + (this.getSource() != null ? this.getSource().getPointerKey() : "null")
                      + ".");

              // `None` in a shape arg (e.g., `shape=[None, 4]`) is the dynamic-dim marker.
              // https://github.com/wala/ML/issues/545: emit `DynamicDim.INSTANCE` instead of raw
              // `null`.
              Dimension<?> dimension =
                  (shapeValue != null)
                      ? new NumericDim(shapeValue.intValue())
                      : DynamicDim.INSTANCE;

              LOGGER.fine("Adding dimension: " + dimension + ".");
              tensorDimensions.add(dimension);
            } else if (instanceFieldIK instanceof AllocationSiteInNode) {
              AllocationSiteInNode innerAsin = (AllocationSiteInNode) instanceFieldIK;
              TypeReference innerReference = innerAsin.concreteType().getReference();

              if (innerReference.equals(tuple)
                  || innerReference.equals(list)
                  || innerReference.equals(TensorFlowTypes.TENSOR_SPEC)
                  || innerReference.equals(TensorFlowTypes.RAGGED_TENSOR_SPEC)) {
                // Nested tuple/list or Spec. Recurse.
                Set<List<Dimension<?>>> nestedShapes =
                    this.getShapesFromShapeArgument(
                        builder, Collections.singleton(instanceFieldIK));

                if (nestedShapes == null) return null;
                for (List<Dimension<?>> nestedShape : nestedShapes)
                  tensorDimensions.add(new CompoundDim(nestedShape));
              } else {
                // Nested element of an unrecognized form; the shape's structure isn't statically
                // resolvable, so return ⊤ rather than aborting (wala/ML#471).
                LOGGER.fine(
                    "Unrecognized nested shape element for instance field: "
                        + describe(pointerKeyForInstanceField)
                        + ", got: "
                        + describe(instanceFieldIK)
                        + "; treating the shape as unknown (⊤).");
                return null;
              }
            } else {
              LOGGER.fine(
                  "Unrecognized shape element for instance field: "
                      + describe(pointerKeyForInstanceField)
                      + ", got: "
                      + describe(instanceFieldIK)
                      + "; treating the shape as unknown (⊤).");
              return null;
            }
          }

          LOGGER.fine(
              "Found possible shape dimensions: "
                  + tensorDimensions
                  + " for field: "
                  + describe(pointerKeyForInstanceField)
                  + " for source: "
                  + describe(this.getSource())
                  + ".");

          // Add the shape dimensions.
          assert possibleDimensions[fieldIndex] == null
              : "Duplicate field index: "
                  + fieldIndex
                  + " in object catalog: "
                  + objectCatalogPointsToSet
                  + ".";

          possibleDimensions[fieldIndex] = tensorDimensions;
          LOGGER.fine(
              "Added shape dimensions: "
                  + tensorDimensions
                  + " for field index: "
                  + fieldIndex
                  + ".");
        }

        // Build the Cartesian product of dimension possibilities across all positions. Empty
        // positions (where `possibleDimensions[k]` has no resolved constant, e.g., a non-literal
        // `BATCH_SIZE` in `tf.random.normal([BATCH_SIZE, 100])`) contribute
        // `UnresolvedDim.INSTANCE` as the single fallback option — wala/ML#721 (previously
        // `DynamicDim`, https://github.com/wala/ML/issues/545). The prior
        // implementation iterated each
        // position's set but only retained the last iterated element, producing a
        // non-deterministic single shape per `i` rather than the full product.
        List<Set<Dimension<?>>> resolved = new ArrayList<>(possibleDimensions.length);
        for (int k = 0; k < possibleDimensions.length; k++) {
          Set<Dimension<?>> s = possibleDimensions[k];
          if (s != null && !s.isEmpty()) {
            resolved.add(s);
            continue;
          }
          // An empty position holds no resolved constant. Before degrading, try to fold a binary
          // op over constant-valued operands (e.g. `self.heads * self.out_features`) via the
          // analysis. `interpretAsInt` in `TensorType.shapeArg` only handles pure-literal source
          // text; this generator-side path resolves field reads and globals through the PTS,
          // reconciling the two shape-argument-extraction paths (wala/ML#581). An unfoldable
          // position is a fixed runtime size the analysis could not compute — `UnresolvedDim`,
          // not `DynamicDim`; an explicit `None` resolves through the constant path above
          // (wala/ML#721).
          Dimension<?> folded = foldArithmeticShapeDim(builder, asin, k);
          resolved.add(Collections.singleton(folded != null ? folded : UnresolvedDim.INSTANCE));
        }

        List<List<Dimension<?>>> shapes = new ArrayList<>();
        shapes.add(new ArrayList<>());
        for (Set<Dimension<?>> options : resolved) {
          List<List<Dimension<?>>> next = new ArrayList<>(shapes.size() * options.size());
          for (List<Dimension<?>> prefix : shapes) {
            for (Dimension<?> d : options) {
              List<Dimension<?>> extended = new ArrayList<>(prefix);
              extended.add(d);
              next.add(extended);
            }
          }
          shapes = next;
        }
        ret.addAll(shapes);
      } else if (asin.getNode()
          .getMethod()
          .getDeclaringClass()
          .getReference()
          .equals(CONSTANT.getDeclaringClass())) {
        // We have a `tf.constant(...)` result. Detect this by checking the
        // *containing method* of the allocation (the `tensorflow/functions/constant.do()`
        // CGNode), not the alloc's concrete type — the latter is the function-name
        // duplicate `Ltensorflow/python/framework/constant_op/constant`, which is a
        // load-bearing function-type signal we want to be able to migrate to canonical
        // `Ltensorflow/python/framework/ops/Tensor` later (wala/ML#459 PR 2). Reading
        // the containing method's declaring class instead decouples this recognition
        // from the alloc-class convention; the two are functionally equivalent today
        // because the only place that allocates `CONSTANT_OP_CONSTANT` is inside
        // `constant.do()`.
        //
        // We need the user-supplied value PTS to recurse into. The XML no longer
        // binds it to the alloc's `value` field (wala/ML#451 reopen — that binding
        // caused Hybridize's `Function.containsPrimitive` to traverse the alloc's
        // fields and find a primitive `ConstantKey` for `tf.constant(N)`-style calls,
        // classifying receiving parameters as primitive). Fall back to walking back
        // from the alloc's `do` CGNode to its calling sites and unioning each call's
        // value-arg PTS. Use the already-unwrapped {@code asin} from the loop header so
        // wrapping {@link InstanceKey}s (e.g. {@link
        // com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKey}) route through the
        // CG-walk just like raw {@link AllocationSiteInNode}s.
        OrdinalSet<InstanceKey> valuePts = getConstantCallValueArgPTS(asin, builder);
        if (valuePts == null || valuePts.isEmpty()) {
          // Defensive fallback in case the `value` field happens to be bound
          // in some other path (e.g. a future XML model that re-introduces
          // it for a sibling endpoint).
          IField valueField =
              builder.getClassHierarchy().resolveField(TensorFlowTypes.CONSTANT_VALUE);
          PointerKey valuePK = builder.getPointerKeyForInstanceField(asin, valueField);
          valuePts = pointerAnalysis.getPointsToSet(valuePK);
        }
        if (valuePts == null || valuePts.isEmpty()) return null;
        Set<List<Dimension<?>>> constantShapes = this.getShapesFromShapeArgument(builder, valuePts);
        if (constantShapes == null) return null;
        ret.addAll(constantShapes);
      } else if (reference.equals(TensorFlowTypes.TENSOR_SPEC)
          || reference.equals(TensorFlowTypes.RAGGED_TENSOR_SPEC)) {
        // We have a TensorSpec or RaggedTensorSpec. These objects carry shape and dtype
        // information in their fields. We extract the 'shape' field and recurse to
        // parse the actual shape structure (usually a tuple or list of integers).
        IField shapeField =
            builder
                .getClassHierarchy()
                .resolveField(
                    reference.equals(TensorFlowTypes.TENSOR_SPEC)
                        ? TensorFlowTypes.SPEC_SHAPE
                        : TensorFlowTypes.RAGGED_SPEC_SHAPE);
        PointerKey shapePK = builder.getPointerKeyForInstanceField(instanceKey, shapeField);
        OrdinalSet<InstanceKey> shapePts = pointerAnalysis.getPointsToSet(shapePK);
        if (shapePts == null || shapePts.isEmpty()) return null;
        Set<List<Dimension<?>>> specShapes = this.getShapesFromShapeArgument(builder, shapePts);
        if (specShapes == null) return null;
        ret.addAll(specShapes);
      } else {
        // Unrecognized top-level shape form — e.g. a runtime tensor such as the result of
        // `tf.shape(y)`, an opaque builder value, or any type that isn't a list/tuple,
        // `tf.constant`,
        // `TensorSpec`, or `RaggedTensorSpec`. Return ⊤ ("tensor of unknown shape") rather than
        // aborting the analysis on otherwise-valid programs (wala/ML#471).
        LOGGER.fine(
            "Unrecognized shape argument form: "
                + reference
                + " for source: "
                + this.getSource()
                + "; treating the shape as unknown (⊤).");
        return null;
      }
    }

    return ret;
  }

  /**
   * Returns the integer element index for an object-catalog key, or {@code null} when the key is
   * not an integer index. A Python list/tuple's object catalog accumulates <em>every</em> attribute
   * name accessed on the allocation, not just its integer element indices &mdash; e.g. a
   * method-name field like {@code "read_data"} left by a virtual-dispatch read on an aliased
   * receiver (<a href="https://github.com/wala/ML/issues/603">wala/ML#603</a>, root cause <a
   * href="https://github.com/wala/ML/issues/608">wala/ML#608</a>). Such non-integer keys are not
   * element indices, so this returns {@code null} and callers skip them. (Previously threw {@link
   * IllegalStateException}, treating a non-integer key as an invariant violation; it isn't one.)
   *
   * @param constantKey The object-catalog key.
   * @return The integer element index, or {@code null} if the key is not an integer index.
   */
  protected static Integer getFieldIndex(ConstantKey<?> constantKey) {
    Object constantKeyValue = constantKey.getValue();

    if (constantKeyValue instanceof Integer) return (Integer) constantKeyValue;
    if (constantKeyValue instanceof String) {
      try {
        return Integer.parseInt((String) constantKeyValue);
      } catch (NumberFormatException e) {
        // A non-integer attribute key (e.g. a method-name field), not an element index. See
        // wala/ML#603.
        return null;
      }
    }

    return null;
  }

  /**
   * Counts the integer-indexed entries of an object catalog, ignoring non-integer attribute keys
   * (see {@link #getFieldIndex}). This is the element count of the underlying list/tuple, used as
   * the leading dimension; the raw catalog size would be inflated by attribute-name pollution (<a
   * href="https://github.com/wala/ML/issues/603">wala/ML#603</a>).
   *
   * @param objectCatalog The object-catalog points-to set.
   * @return The number of integer-indexed (element) entries.
   */
  protected static int integerCatalogSize(OrdinalSet<InstanceKey> objectCatalog) {
    int count = 0;
    for (InstanceKey ik : objectCatalog)
      if (ik instanceof ConstantKey && getFieldIndex((ConstantKey<?>) ik) != null) count++;
    return count;
  }

  /**
   * Attempts to fold the shape dimension at {@code fieldIndex} of the shape list/tuple allocated at
   * {@code listAsin} when that element is a binary op over constant-valued operands (e.g. {@code
   * self.heads * self.out_features}).
   *
   * <p>The element write is located in the list's allocating node IR, and the binary op's operands
   * are resolved to constants through the points-to analysis (so field reads and globals resolve,
   * not just source-text literals). This is the generator-side half of the shape-argument
   * reconciliation in wala/ML#581: {@link TensorType#shapeArg(CGNode, int,
   * PropagationCallGraphBuilder)} can fold literal source text via the embedded interpreter but has
   * no points-to analysis to resolve {@code self.X}.
   *
   * @param builder The propagation call graph builder, used to resolve operand points-to sets.
   * @param listAsin The allocation site of the shape list/tuple.
   * @param fieldIndex The position whose dimension is being folded.
   * @return A {@link NumericDim} holding the folded value, or {@code null} if the element is not a
   *     constant-foldable binary op.
   */
  private Dimension<?> foldArithmeticShapeDim(
      PropagationCallGraphBuilder builder, AllocationSiteInNode listAsin, int fieldIndex) {
    CGNode node = listAsin.getNode();
    if (node.getIR() == null) return null;
    DefUse du = node.getDU();
    SymbolTable st = node.getIR().getSymbolTable();
    SSANewInstruction allocInstr = node.getIR().getNew(listAsin.getSite());
    if (allocInstr == null) return null;
    int listVn = allocInstr.getDef();

    for (Iterator<SSAInstruction> uses = du.getUses(listVn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      int objRef;
      int writtenVal;
      Integer index = null;

      if (use instanceof PythonPropertyWrite) {
        PythonPropertyWrite w = (PythonPropertyWrite) use;
        objRef = w.getObjectRef();
        writtenVal = w.getValue();
        int memberVn = w.getMemberRef();
        if (st.isNumberConstant(memberVn))
          index = ((Number) st.getConstantValue(memberVn)).intValue();
        else if (st.isStringConstant(memberVn)) {
          try {
            index = Integer.parseInt(st.getStringValue(memberVn));
          } catch (NumberFormatException e) {
            // not an index write.
          }
        }
      } else if (use instanceof SSAPutInstruction) {
        SSAPutInstruction p = (SSAPutInstruction) use;
        if (p.isStatic()) continue;
        objRef = p.getRef();
        writtenVal = p.getVal();
        try {
          index = Integer.parseInt(p.getDeclaredField().getName().toString());
        } catch (NumberFormatException e) {
          // not an index write.
        }
      } else continue;

      if (objRef != listVn || index == null || index.intValue() != fieldIndex) continue;

      // Shared with `TensorType.shapeArg` so the two shape-argument paths agree (wala/ML#581).
      Dimension<?> folded = TensorType.foldArithmeticDim(builder, node, st, du, writtenVal);
      if (folded != null) return folded;
      folded = this.prodOfShapeVectorDim(builder, node, st, writtenVal);
      if (folded != null) return folded;
      folded = this.resolveShapeVectorElementDim(builder, node, st, writtenVal);
      if (folded != null) return folded;
      folded = this.resolveArithmeticOverSubscriptDim(builder, node, st, writtenVal);
      if (folded != null) return folded;
      return this.resolveStoredAttributeDim(builder, node, writtenVal);
    }
    return null;
  }

  /**
   * Resolves a constant-indexed subscript of a shape vector (e.g. {@code input_shape[2]}) to the
   * denoted dimension, with Python negative-index semantics (wala/ML#712).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number of the candidate subscript result.
   * @return The subscripted {@link Dimension}, or {@code null} when the value isn't a
   *     constant-indexed subscript of a resolvable shape vector or the index is out of range.
   */
  Dimension<?> resolveShapeVectorElementDim(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0) return null;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof PythonPropertyRead)) return null;
    PythonPropertyRead read = (PythonPropertyRead) def;
    int memberVn = read.getMemberRef();
    Integer index;
    if (st.isStringConstant(memberVn)) {
      // Subscript indices can surface as string constants (like the index writes above); a
      // non-numeric string is an attribute read, not a subscript.
      try {
        index = Integer.parseInt(st.getStringValue(memberVn));
      } catch (NumberFormatException e) {
        return null;
      }
    } else {
      index = constantIntValueOrSentinel(builder, node, st, memberVn);
      if (index == null || Objects.equals(index, UNRESOLVED_BOUND)) return null;
    }

    Set<List<Dimension<?>>> shapes =
        this.getShapesOfShapeVector(builder, node, read.getObjectRef());
    if (shapes == null || shapes.size() != 1) return null; // Ambiguous ranks: not resolvable.
    List<Dimension<?>> dims = shapes.iterator().next();
    int k = index < 0 ? dims.size() + index : index;
    if (k < 0 || k >= dims.size()) return null;
    return dims.get(k);
  }

  /**
   * Folds an arithmetic expression whose operands may be shape-vector subscripts (e.g. {@code
   * int(input_shape[2] / self.num_attention_heads)}, NLPGNN's {@code ALBERTAttention.build}) to a
   * {@link NumericDim} (wala/ML#714). An {@code int(...)} builtin wrapper is peeled first; a
   * division only folds when the wrapper marked the truncation or the division is exact, since a
   * bare {@code /} is Python float division.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number of the candidate expression result.
   * @return The folded {@link NumericDim}, or {@code null} when the expression doesn't resolve.
   */
  private Dimension<?> resolveArithmeticOverSubscriptDim(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0 || node.getDU() == null) return null;
    int peeled = peelIntBuiltin(builder, node, vn);
    boolean truncated = peeled != vn;
    SSAInstruction def = node.getDU().getDef(peeled);

    if (def instanceof SSABinaryOpInstruction) {
      SSABinaryOpInstruction binOp = (SSABinaryOpInstruction) def;
      Integer left = this.scalarOperandOrNull(builder, node, st, binOp.getUse(0));
      Integer right = this.scalarOperandOrNull(builder, node, st, binOp.getUse(1));
      if (left == null || right == null) {
        // When exactly one operand resolves and that operand derives from a shape vector, the
        // expression is dimension arithmetic and the result is one scalar dimension whose value
        // is unknown, so the value degrades rather than the element (wala/ML#717). NLPGNN's
        // `input_shape[-1] * self.embedding_size` with a config-sourced factor is the witness:
        // a fixed runtime size the analysis could not compute, i.e. `UnresolvedDim`, not
        // `DynamicDim` (wala/ML#721).
        if ((left == null) != (right == null)
            && this.derivesFromShapeVector(
                builder, node, st, left == null ? binOp.getUse(1) : binOp.getUse(0)))
          return UnresolvedDim.INSTANCE;
        return null;
      }
      if (binOp.getOperator() == IBinaryOpInstruction.Operator.DIV
          && !truncated
          && (right == 0 || left % right != 0)) return null; // Non-exact bare float division.
      Integer value = TensorType.applyIntBinOp(binOp.getOperator(), left, right);
      return value == null ? null : new NumericDim(value);
    }

    // A bare int(x) wrapper over a directly resolvable scalar.
    if (truncated) {
      Dimension<?> dim = this.resolveShapeVectorElementDim(builder, node, st, peeled);
      if (dim != null) return dim;
      return this.prodOfShapeVectorDim(builder, node, st, peeled);
    }
    return null;
  }

  /**
   * Decides whether a value derives from a shape vector: a shape-vector subscript (single- or
   * multi-member) or an {@code np.prod} fold. Evidence that arithmetic over the value is dimension
   * arithmetic (wala/ML#717).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number to test.
   * @return {@code true} iff the value resolves through a shape-vector path.
   */
  private boolean derivesFromShapeVector(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    int peeled = peelIntBuiltin(builder, node, vn);
    if (this.resolveShapeVectorElementDim(builder, node, st, peeled) != null) return true;
    if (this.resolveShapeVectorElementDims(builder, node, st, peeled) != null) return true;
    return this.prodOfShapeVectorDim(builder, node, st, peeled) != null;
  }

  /**
   * Resolves an arithmetic operand to a constant integer: a points-to constant, a numeric
   * shape-vector subscript, an {@code np.prod} fold, or a nested arithmetic expression
   * (wala/ML#714).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The operand's value number.
   * @return The operand's integer value, or {@code null} when it doesn't resolve.
   */
  private Integer scalarOperandOrNull(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    int peeled = peelIntBuiltin(builder, node, vn);
    Integer constant = constantIntValueOrSentinel(builder, node, st, peeled);
    if (constant != null && !Objects.equals(constant, UNRESOLVED_BOUND)) return constant;
    Dimension<?> dim = this.resolveShapeVectorElementDim(builder, node, st, peeled);
    if (dim == null) dim = this.prodOfShapeVectorDim(builder, node, st, peeled);
    if (dim == null) dim = this.resolveArithmeticOverSubscriptDim(builder, node, st, peeled);
    return dim instanceof NumericDim ? ((NumericDim) dim).value() : null;
  }

  /**
   * Resolves a single-element list/tuple allocation to its constant integer element (e.g. the
   * {@code axis=[-1]} form of {@code tf.expand_dims}, as in NLPGNN's {@code WDEmbedding.call}).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for PA lookup.
   * @param ik The candidate container {@link InstanceKey}.
   * @return The single element's constant value, or {@code null} when the key isn't a
   *     single-element list/tuple allocation of a resolvable constant (wala/ML#714).
   */
  protected static Integer singleElementConstant(
      PropagationCallGraphBuilder builder, InstanceKey ik) {
    AllocationSiteInNode asin = getAllocationSiteInNode(ik);
    if (asin == null) return null;
    TypeReference type = asin.getSite().getDeclaredType();
    if (!type.equals(list) && !type.equals(tuple)) return null;
    CGNode node = asin.getNode();
    if (node.getIR() == null || node.getDU() == null) return null;
    SSANewInstruction alloc = node.getIR().getNew(asin.getSite());
    if (alloc == null) return null;
    int containerVn = alloc.getDef();
    SymbolTable st = node.getIR().getSymbolTable();
    Integer element = null;
    for (Iterator<SSAInstruction> uses = node.getDU().getUses(containerVn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      int writtenVal;
      Integer index = null;
      if (use instanceof PythonPropertyWrite) {
        PythonPropertyWrite write = (PythonPropertyWrite) use;
        if (write.getObjectRef() != containerVn) continue;
        int memberVn = write.getMemberRef();
        if (st.isNumberConstant(memberVn))
          index = ((Number) st.getConstantValue(memberVn)).intValue();
        else if (st.isStringConstant(memberVn)) {
          try {
            index = Integer.parseInt(st.getStringValue(memberVn));
          } catch (NumberFormatException e) {
            continue;
          }
        }
        writtenVal = write.getValue();
      } else if (use instanceof SSAPutInstruction && !((SSAPutInstruction) use).isStatic()) {
        SSAPutInstruction put = (SSAPutInstruction) use;
        if (put.getRef() != containerVn) continue;
        try {
          index = Integer.parseInt(put.getDeclaredField().getName().toString());
        } catch (NumberFormatException e) {
          continue;
        }
        writtenVal = put.getVal();
      } else continue;
      if (index == null) continue;
      if (index != 0 || element != null) return null; // Not a single-element container.
      Integer constant = constantIntValueOrSentinel(builder, node, st, writtenVal);
      if (constant == null || Objects.equals(constant, UNRESOLVED_BOUND)) return null;
      element = constant;
    }
    return element;
  }

  /**
   * Applies the explicit {@code model.build(input_shape=...)} contract seed when the given value is
   * the first input parameter of a Keras {@code call} body (wala/ML#717).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code valueNumber}.
   * @param valueNumber The value number being resolved.
   * @return The declared contract shapes, or {@code null} when the value isn't a {@code call} input
   *     or no contract resolves.
   */
  private Set<List<Dimension<?>>> contractSeedForCallInput(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    if (node.getDU() == null || node.getIR() == null) return null;
    if (node.getDU().getDef(valueNumber) != null) return null; // Not a parameter.
    if (node.getMethod().isStatic() || node.getIR().getNumberOfParameters() < 3) return null;
    if (node.getIR().getParameter(2) != valueNumber) return null; // Not the first input.

    // The recognizer scans the whole call graph and the result depends only on the call body's
    // class, so it is memoized per class. A sentinel is seeded before computing: the contract's
    // literal resolution can re-enter the seed through the shape machinery, and the sentinel
    // floors the re-entry to "no contract" (sound) instead of recursing.
    Map<IClass, Optional<Set<List<Dimension<?>>>>> cache =
        BUILD_CONTRACT_CACHE.computeIfAbsent(
            builder, b -> Collections.synchronizedMap(new HashMap<>()));
    IClass key = node.getMethod().getDeclaringClass();
    Optional<Set<List<Dimension<?>>>> memo = cache.get(key);
    if (memo == null) {
      cache.put(key, Optional.empty());
      Set<List<Dimension<?>>> computed = this.explicitBuildContractShapes(builder, node);
      LOGGER.fine(
          () ->
              "Contract seed consulted for vn "
                  + valueNumber
                  + " of "
                  + describe(node)
                  + " -> "
                  + computed
                  + ".");
      memo = Optional.ofNullable(computed);
      cache.put(key, memo);
    }
    return memo.orElse(null);
  }

  /**
   * Resolves the input contract an explicitly invoked {@code model.build(input_shape=(...))}
   * declares for the given Keras {@code call} body (wala/ML#717). The Keras contract ties a built
   * layer's call inputs to the built shape, so when the runtime arguments do not resolve, the
   * declared literal is a sound seed for the {@code call} input's shape. Only non-trampoline
   * (user-written) {@code build} invocations count; the lazy-build trampoline passes no shape.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param callNode The code body of the Keras {@code call} whose input is being resolved.
   * @return The union of the declared shapes across explicit {@code build} callers, or {@code null}
   *     when there is none or none resolves.
   */
  private Set<List<Dimension<?>>> explicitBuildContractShapes(
      PropagationCallGraphBuilder builder, CGNode callNode) {
    String className = callNode.getMethod().getDeclaringClass().getReference().getName().toString();
    String callSuffix = "/" + CALLABLE_METHOD_NAME_FOR_KERAS_MODELS;
    if (!className.endsWith(callSuffix)) return null;
    String pythonClassName = className.substring(0, className.length() - callSuffix.length());

    // The receiving class commonly defines no `build` of its own (the explicit call hits Keras's
    // base implementation, which the analysis does not model), so the contract is recognized
    // syntactically: an invoke whose callable is a `build` attribute read off an instance of the
    // class, mirroring the `set_shape` recognizer.
    Set<List<Dimension<?>>> combined = null;
    for (CGNode caller : builder.getCallGraph()) {
      if (caller.getIR() == null || caller.getDU() == null) continue;
      if (!(caller.getMethod() instanceof AstMethod)) continue;
      SymbolTable st = caller.getIR().getSymbolTable();
      for (Iterator<SSAInstruction> it = caller.getIR().iterateAllInstructions(); it.hasNext(); ) {
        SSAInstruction inst = it.next();
        if (!(inst instanceof PythonInvokeInstruction)) continue;
        PythonInvokeInstruction call = (PythonInvokeInstruction) inst;
        if (call.getNumberOfUses() < 1) continue;
        SSAInstruction targetDef = caller.getDU().getDef(call.getUse(0));
        if (!(targetDef instanceof PythonPropertyRead)) continue;
        PythonPropertyRead prop = (PythonPropertyRead) targetDef;
        int memberVn = prop.getMemberRef();
        if (!st.isStringConstant(memberVn)
            || !KERAS_BUILD_METHOD_NAME.equals(st.getStringValue(memberVn))) continue;

        // The receiver must be an instance of the `call` body's class.
        PointerKey receiverKey =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(caller, prop.getObjectRef());
        boolean receiverMatches = false;
        for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(receiverKey)) {
          AllocationSiteInNode asin = getAllocationSiteInNode(ik);
          if (asin != null
              && asin.getNode()
                  .getMethod()
                  .getDeclaringClass()
                  .getReference()
                  .getName()
                  .toString()
                  .equals(pythonClassName)) {
            receiverMatches = true;
            break;
          }
        }
        if (!receiverMatches) continue;

        int argVn = call.getUse("input_shape");
        if (argVn <= 0 && call.getNumberOfUses() >= 2) argVn = call.getUse(1);
        if (argVn <= 0) continue;
        Set<List<Dimension<?>>> declared =
            this.shapesOfShapeVectorOrLiteralList(builder, caller, argVn, HashSetFactory.make());
        int finalArgVn = argVn;
        LOGGER.fine(
            () ->
                "Explicit build contract for "
                    + pythonClassName
                    + " found in "
                    + describe(caller)
                    + " at "
                    + call
                    + " with input_shape vn "
                    + finalArgVn
                    + " -> "
                    + declared
                    + ".");
        if (declared == null || declared.isEmpty()) continue;
        if (combined == null) combined = HashSetFactory.make();
        combined.addAll(declared);
      }
    }
    return combined;
  }

  /**
   * Peels an {@code int(...)} builtin wrapper: for an invoke of the {@code int} builtin with a
   * single positional argument, returns the argument's value number; otherwise returns {@code vn}
   * unchanged (wala/ML#714).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the target.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to peel.
   * @return The wrapped argument's value number, or {@code vn} when it isn't an {@code int(x)}
   *     call.
   */
  private static int peelIntBuiltin(PropagationCallGraphBuilder builder, CGNode node, int vn) {
    if (vn <= 0 || node.getDU() == null) return vn;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof PythonInvokeInstruction)) return vn;
    PythonInvokeInstruction invoke = (PythonInvokeInstruction) def;
    if (invoke.getNumberOfUses() != 2) return vn; // Exactly `int(x)`: the callable plus one arg.
    Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
    if (targets == null || targets.isEmpty()) return vn;
    for (CGNode target : targets)
      if (!target
          .getMethod()
          .getDeclaringClass()
          .getReference()
          .getName()
          .toString()
          .equals("Lwala/builtin/int")) return vn;
    return invoke.getUse(1);
  }

  /**
   * Resolves a bare instance-attribute read off the enclosing method's {@code self} (e.g. {@code
   * self.hidden_size}) to a dimension by chasing the attribute's write sites in the receiver
   * class's method bodies and resolving each stored value in its defining node. Covers {@code
   * build}-computed sizes such as {@code self.hidden_size = input_shape[2]}, whose stored value has
   * an empty points-to set, so the constant-propagation path cannot see it (wala/ML#712).
   *
   * <p>The chase is depth-one by construction: a stored value that is itself another bare attribute
   * read does not resolve.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number of the candidate attribute read.
   * @return The stored value's dimension, or {@code null} when the value isn't a {@code self}
   *     attribute read, no write site resolves, or distinct write sites resolve to different
   *     dimensions.
   */
  private Dimension<?> resolveStoredAttributeDim(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    if (vn <= 0 || node.getDU() == null || node.getIR() == null) return null;
    SSAInstruction def = node.getDU().getDef(vn);
    // A user attribute read is always a PythonPropertyRead: the front end emits
    // SSAGetInstruction only for the inherited-member propagation of a class declaration, whose
    // temporaries never surface as chased stored values (wala/ML#725).
    if (!(def instanceof PythonPropertyRead)) return null;

    PythonPropertyRead read = (PythonPropertyRead) def;
    SymbolTable st = node.getIR().getSymbolTable();
    int memberVn = read.getMemberRef();
    if (!st.isStringConstant(memberVn)) return null;
    String attributeName = st.getStringValue(memberVn);
    int objectVn = read.getObjectRef();

    // Chase reads off the enclosing method's `self` through the receiver class's method bodies;
    // for any other local object, chase the write sites in the same code body (e.g. a
    // module-level `param.batch_size = 8` feeding `model.build(input_shape=(3, param.batch_size,
    // param.maxlen))`, wala/ML#717). The object's allocation site is commonly shared across
    // scripts (a helper-allocated parameter holder under one calling context), so the abstract
    // field unions every script's write and the constant-propagation path sees only the
    // ambiguous merge; the same-body write is the one that holds at the read.
    if (node.getMethod().getNumberOfParameters() < 2 || objectVn != node.getIR().getParameter(1))
      return this.resolveLocalObjectAttributeDim(builder, node, objectVn, attributeName);

    PointerKey selfPk =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, objectVn);
    Dimension<?> found = null;
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(selfPk)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) return null;
      // A user Python object allocates as `Lobject` inside its class's synthetic constructor, so
      // the class identity lives in the allocating node's declaring class.
      IClass pythonClass = asin.getNode().getMethod().getDeclaringClass();
      Map<Pair<IClass, String>, Optional<Dimension<?>>> cache =
          STORED_ATTRIBUTE_CACHE.computeIfAbsent(
              builder, b -> Collections.synchronizedMap(new HashMap<>()));
      Pair<IClass, String> key = Pair.make(pythonClass, attributeName);
      Optional<Dimension<?>> memo = cache.get(key);
      if (memo == null) {
        // Seed an empty sentinel before computing: the chase can re-enter itself on the same key
        // through the shape machinery (a self-referential attribute), and the sentinel floors the
        // re-entry to "unresolved" (sound) instead of recursing.
        cache.put(key, Optional.empty());
        Dimension<?> chased =
            this.resolveStoredAttributeDimInClass(builder, pythonClass, attributeName);
        LOGGER.fine(
            () ->
                "resolveStoredAttributeDim: "
                    + pythonClass
                    + "."
                    + attributeName
                    + " resolved to "
                    + chased);
        memo = Optional.ofNullable(chased);
        cache.put(key, memo);
      }
      Dimension<?> stored = memo.orElse(null);
      if (stored == null) return null;
      if (found != null && !found.equals(stored)) return null; // Conflicting receivers.
      found = stored;
    }
    return found;
  }

  /**
   * Chases the write sites of {@code attributeName} on {@code self} across the given Python class's
   * method-body nodes and resolves each stored value in its defining node.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param pythonClass The Python class whose method bodies are scanned.
   * @param attributeName The attribute whose writes are chased.
   * @return The stored value's dimension, or {@code null} when no write resolves or distinct writes
   *     resolve to different dimensions.
   */
  private Dimension<?> resolveStoredAttributeDimInClass(
      PropagationCallGraphBuilder builder, IClass pythonClass, String attributeName) {
    String classPrefix = pythonClass.getReference().getName().toString() + "/";
    Dimension<?> found = null;
    for (CGNode candidate : builder.getCallGraph()) {
      if (candidate.getIR() == null || candidate.getDU() == null) continue;
      IMethod method = candidate.getMethod();
      if (!(method instanceof AstMethod)) continue; // Only real method bodies hold user writes.
      String declaringClassName = method.getDeclaringClass().getReference().getName().toString();
      if (!declaringClassName.startsWith(classPrefix)) continue;
      if (method.getNumberOfParameters() < 2) continue;
      int selfVn = candidate.getIR().getParameter(1);
      SymbolTable st = candidate.getIR().getSymbolTable();

      for (SSAInstruction inst : candidate.getIR().getInstructions()) {
        int storedVn = -1;
        // A `self` attribute write is always a PythonPropertyWrite: the front end emits
        // SSAPutInstruction only for module, function-binding, and class-declaration member
        // writes, whose receiver is the script or class object and can never be `self` (a method
        // body declaring a nested config class carries such puts, and they must be skipped, which
        // `tf2_test_build_stored_attr.py`'s nested-class cases pin). See wala/ML#725.
        if (inst instanceof PythonPropertyWrite) {
          PythonPropertyWrite write = (PythonPropertyWrite) inst;
          int memberVn = write.getMemberRef();
          if (write.getObjectRef() == selfVn
              && st.isStringConstant(memberVn)
              && attributeName.equals(st.getStringValue(memberVn))) storedVn = write.getValue();
        }
        if (storedVn <= 0) continue;

        Dimension<?> stored = this.resolveStoredValueDim(builder, candidate, st, storedVn);
        if (stored == null) return null; // An unresolvable write makes the attribute unresolvable.
        if (found != null && !found.equals(stored)) return null; // Conflicting writes.
        found = stored;
      }
    }
    return found;
  }

  /**
   * Chases the write sites of {@code attributeName} on the given local object within the same code
   * body and resolves each stored value (wala/ML#717). Covers NLPGNN's entry idiom, where a
   * module-level {@code param.batch_size = 8} feeds {@code model.build(input_shape=(3,
   * param.batch_size, param.maxlen))} but {@code param}'s helper-allocated holder is one abstract
   * object shared by every entry script, so the field's points-to set is the ambiguous union of all
   * scripts' writes. The chase is same-body and same-value-number by construction: it recovers the
   * per-script value that holds at the read, and writes through other references to the object are
   * not seen.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose code body holds the read and the chased writes.
   * @param objectVn The value number of the object whose attribute writes are chased.
   * @param attributeName The attribute whose writes are chased.
   * @return The stored value's dimension, or {@code null} when no write resolves or distinct writes
   *     resolve to different dimensions.
   */
  private Dimension<?> resolveLocalObjectAttributeDim(
      PropagationCallGraphBuilder builder, CGNode node, int objectVn, String attributeName) {
    SymbolTable st = node.getIR().getSymbolTable();
    Dimension<?> found = null;
    for (Iterator<SSAInstruction> uses = node.getDU().getUses(objectVn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      int storedVn = -1;
      // A user attribute write is always a PythonPropertyWrite: the front end emits
      // SSAPutInstruction only for module, function-binding, and class-declaration member writes,
      // whose receiver is the script or class object rather than a chased local (wala/ML#725).
      if (use instanceof PythonPropertyWrite) {
        PythonPropertyWrite write = (PythonPropertyWrite) use;
        int memberVn = write.getMemberRef();
        if (write.getObjectRef() == objectVn
            && st.isStringConstant(memberVn)
            && attributeName.equals(st.getStringValue(memberVn))) storedVn = write.getValue();
      }
      if (storedVn <= 0) continue;

      Dimension<?> stored = this.resolveStoredValueDim(builder, node, st, storedVn);
      if (stored == null) return null; // An unresolvable write makes the attribute unresolvable.
      if (found != null && !found.equals(stored)) return null; // Conflicting writes.
      found = stored;
    }
    return found;
  }

  /**
   * Resolves a stored attribute value to a dimension in its defining node: an arithmetic fold, an
   * {@code np.prod} over a shape vector, a shape-vector subscript, arithmetic over a subscript, or
   * a constant.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code storedVn}.
   * @param st The node's symbol table.
   * @param storedVn The stored value's value number.
   * @return The stored value's dimension, or {@code null} when it doesn't resolve.
   */
  private Dimension<?> resolveStoredValueDim(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int storedVn) {
    Dimension<?> stored = TensorType.foldArithmeticDim(builder, node, st, node.getDU(), storedVn);
    if (stored == null) stored = this.prodOfShapeVectorDim(builder, node, st, storedVn);
    if (stored == null) stored = this.resolveShapeVectorElementDim(builder, node, st, storedVn);
    if (stored == null)
      stored = this.resolveArithmeticOverSubscriptDim(builder, node, st, storedVn);
    if (stored == null) {
      Integer constant = constantIntValueOrSentinel(builder, node, st, storedVn);
      if (constant != null && !Objects.equals(constant, UNRESOLVED_BOUND))
        stored = new NumericDim(constant);
    }
    return stored;
  }

  /**
   * Folds an {@code np.prod(v)} call over a resolvable shape vector into a {@link NumericDim}
   * holding the product of its static dimensions (wala/ML#707). Mirrors NLPGNN's {@code
   * einsum_via_matmul}, where {@code inner_dim = np.prod(input_shape[-num_inner_dims:])} feeds a
   * {@code tf.reshape} target shape.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number of the candidate {@code np.prod} result.
   * @return The folded {@link NumericDim}, or {@code null} when the value isn't an {@code np.prod}
   *     call, its argument doesn't resolve to a single shape, any dimension is non-static, or the
   *     product exceeds the int range.
   */
  private Dimension<?> prodOfShapeVectorDim(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0) return null;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof PythonInvokeInstruction)) return null;
    PythonInvokeInstruction invoke = (PythonInvokeInstruction) def;
    // Fold only the plain `np.prod(v)` form: an extra positional or keyword argument (e.g.
    // `axis=...`) changes the semantics, including the result's rank.
    int numKeywords = invoke.getKeywords() != null ? invoke.getKeywords().size() : 0;
    if (invoke.getNumberOfUses() - numKeywords != 2 || numKeywords != 0) return null;
    SSAInstruction funcDef = node.getDU().getDef(invoke.getUse(0));
    if (!(funcDef instanceof PythonPropertyRead)) return null;
    int memberVn = ((PythonPropertyRead) funcDef).getMemberRef();
    if (!st.isStringConstant(memberVn) || !"prod".equals(st.getStringValue(memberVn))) return null;

    Set<List<Dimension<?>>> shapes = this.getShapesOfShapeVector(builder, node, invoke.getUse(1));
    if (shapes == null || shapes.size() != 1) return null;
    return prodOfDims(shapes.iterator().next());
  }

  /**
   * Folds one shape's dimension product: a fully numeric shape folds to a {@link NumericDim}; a
   * non-numeric member degrades the value rather than the element to null, since the product of a
   * shape vector is always one scalar dimension (wala/ML#718). Degradation taints toward {@link
   * DynamicDim} (wala/ML#721): a product over any {@code Dynamic} factor is itself {@code Dynamic}
   * (the runtime {@code TensorShape} reports {@code None} for arithmetic over a {@code None} axis),
   * while a product over merely unresolved factors — or one that overflows the representable range
   * — is a fixed runtime value the analysis could not compute, {@link UnresolvedDim}.
   *
   * @param dims The shape whose dimensions are folded.
   * @return The folded dimension.
   */
  private static Dimension<?> prodOfDims(List<Dimension<?>> dims) {
    for (Dimension<?> d : dims) if (d instanceof DynamicDim) return DynamicDim.INSTANCE;
    long product = 1;
    for (Dimension<?> d : dims) {
      if (!(d instanceof NumericDim)) return UnresolvedDim.INSTANCE; // Unknown fixed factor.
      try {
        product = Math.multiplyExact(product, ((NumericDim) d).value());
      } catch (ArithmeticException e) {
        return UnresolvedDim.INSTANCE; // The product overflows long: not representable.
      }
      if (product < 0 || product > Integer.MAX_VALUE) return UnresolvedDim.INSTANCE;
    }
    return new NumericDim((int) product);
  }

  /**
   * Multi-member counterpart of {@link #prodOfShapeVectorDim}: one product option per source-shape
   * member (wala/ML#718).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number of the candidate {@code np.prod} result.
   * @return The product options, or {@code null} when the value isn't a plain {@code np.prod} over
   *     a resolvable multi-member shape vector (the singleton path covers size one).
   */
  private Set<Dimension<?>> prodOfShapeVectorDims(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0) return null;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof PythonInvokeInstruction)) return null;
    PythonInvokeInstruction invoke = (PythonInvokeInstruction) def;
    int numKeywords = invoke.getKeywords() != null ? invoke.getKeywords().size() : 0;
    if (invoke.getNumberOfUses() - numKeywords != 2 || numKeywords != 0) return null;
    SSAInstruction funcDef = node.getDU().getDef(invoke.getUse(0));
    if (!(funcDef instanceof PythonPropertyRead)) return null;
    int memberVn = ((PythonPropertyRead) funcDef).getMemberRef();
    if (!st.isStringConstant(memberVn) || !"prod".equals(st.getStringValue(memberVn))) return null;

    Set<List<Dimension<?>>> shapes = this.getShapesOfShapeVector(builder, node, invoke.getUse(1));
    if (shapes == null || shapes.size() < 2) return null; // The singleton path covers size one.
    Set<Dimension<?>> options = HashSetFactory.make();
    for (List<Dimension<?>> dims : shapes) options.add(prodOfDims(dims));
    return options;
  }

  /**
   * Returns the union of {@code value}-argument points-to sets across every caller call site that
   * resolved to the {@code do} {@link CGNode} of the given {@code tf.constant} allocation. Used by
   * {@link #getShapesFromShapeArgument} as a fallback when the alloc's {@code value} field PTS is
   * empty &mdash; which it now always is, since {@code tensorflow.xml} stopped binding the user's
   * value to that field to keep Hybridize's {@code Function.containsPrimitive} from finding a
   * primitive {@link ConstantKey} through the alloc's instance-field chain (wala/ML#451 reopen).
   *
   * <p>The walk is structural: {@code asin.getNode()} is the synthetic {@code constant_op.constant
   * .do} CGNode where the {@code <new>} fired; its CG predecessors are the user-side CGNodes that
   * called {@code tf.constant(...)}. For each such predecessor, every {@link CallSiteReference}
   * resolving to {@code asin.getNode()} contributes the value argument's PTS via {@code
   * call.getUse(1)} (use 0 is the function-object self).
   *
   * @param constAlloc The {@link AllocationSiteInNode} of a {@code Ltensorflow/python/framework/
   *     constant_op/constant} allocation.
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return The union of value-arg points-to sets across all matching caller sites; an empty {@link
   *     OrdinalSet} if the alloc's CGNode has no callers (defensive — shouldn't happen in practice
   *     for a reachable alloc).
   */
  protected OrdinalSet<InstanceKey> getConstantCallValueArgPTS(
      AllocationSiteInNode constAlloc, PropagationCallGraphBuilder builder) {
    CGNode allocNode = constAlloc.getNode();
    CallGraph cg = builder.getCallGraph();
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Set<InstanceKey> all = HashSetFactory.make();
    for (Iterator<CGNode> it = cg.getPredNodes(allocNode); it.hasNext(); ) {
      CGNode caller = it.next();
      com.ibm.wala.ssa.IR callerIR = caller.getIR();
      if (callerIR == null) continue;
      for (SSAInstruction inst : callerIR.getInstructions()) {
        if (!(inst instanceof SSAAbstractInvokeInstruction call)) continue;
        // Check whether this call site resolves to allocNode. Iterating
        // `getPossibleSites(caller, allocNode)` and matching by `CallSiteReference`
        // sometimes returns empty under Python's CG modeling (the trampoline
        // target selector dispatches dynamically); the structural per-instruction
        // walk via `getPossibleTargets` is the reliable check.
        boolean targets = false;
        for (CGNode target : cg.getPossibleTargets(caller, call.getCallSite())) {
          if (target.equals(allocNode)) {
            targets = true;
            break;
          }
        }
        if (!targets) continue;
        // use(0) is the function-object (self); use(1) is the user's `value` argument.
        if (call.getNumberOfUses() < 2) continue;
        int valueVn = call.getUse(1);
        PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(caller, valueVn);
        for (InstanceKey ik : pa.getPointsToSet(pk)) all.add(ik);
      }
    }
    return OrdinalSet.toOrdinalSet(all, pa.getInstanceKeyMapping());
  }

  /**
   * Returns the default shapes when no explicit shape argument is provided. Implementations should
   * return {@code null} when the generator is known to produce a tensor but its shape cannot be
   * determined (unknown / ⊤). An empty set should be returned only when the variable is provably
   * not a tensor (⊥). A non-empty set carries concrete shape information.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The default shapes, or {@code null} if the shape is unknown.
   */
  protected abstract Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder);

  /**
   * Record-carrying counterpart of {@link #getDefaultShapes(PropagationCallGraphBuilder)}
   * (wala/ML#718). The default lifts the legacy result through virtual dispatch, so subclass
   * overrides of {@link #getDefaultShapes} apply and partials merely collapse; generators whose
   * default shape can be partially resolvable override this to keep the members.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The resolution result.
   */
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * Returns the value number for the shape argument in the function call. A return value of a
   * number less than or equal to zero signifies that there is no shape parameter.
   *
   * @return The value number for the shape argument in the function call. May return a number less
   *     than or equal to 0 if there is no shape parameter.
   */
  protected int getShapeArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getShapeParameterPosition());
  }

  /**
   * Returns the position of the shape parameter in the TensorFlow function.
   *
   * @return The position of the shape parameter in the TensorFlow function or a number less than 0
   *     if there is no shape parameter.
   */
  protected abstract int getShapeParameterPosition();

  /**
   * Returns the name of the shape parameter in the TensorFlow function.
   *
   * @return The name of the shape parameter in the TensorFlow function or <code>null</code> if
   *     there is no shape parameter.
   */
  protected abstract String getShapeParameterName();

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return a set of shapes, where each shape is represented as a list of dimensions
   */
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName(), true);
    if (valNum <= 0) return this.getDefaultShapes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());

    // If the argument shape is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultShapes(builder);
    else
      // The shape points-to set is non-empty, meaning that the shape was explicitly set.
      return this.getShapesFromShapeArgument(builder, pointsToSet);
  }

  /**
   * Record-carrying counterpart of {@link #getShapes(PropagationCallGraphBuilder)} (wala/ML#718).
   * The default lifts the legacy result, collapsing nothing; generators whose output shape can be
   * partially resolvable (e.g. {@link Reshape}, whose target may be a partially resolvable shape
   * vector) override this so the resolvable members survive the generator boundary.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The resolution result.
   */
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getShapes(builder));
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator. The shape is inferred
   * from the argument represented by the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valueNumber The value number of the argument from which to infer the shape.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapes(
      PropagationCallGraphBuilder builder, int valueNumber) {
    return getShapes(builder, this.getNode(), valueNumber);
  }

  /**
   * Returns the possible shapes of the tensor represented by the given value number in the
   * specified node. This method uses a multi-staged approach, falling back to interprocedural
   * generator-based tracing if standard points-to analysis fails.
   *
   * <p>Memoized on {@code (node, valueNumber)} per builder by the worklist engine (wala/ML#365) to
   * eliminate redundant recomputation across caller-walk / SSA-DU chain fallback recursion. The PA
   * is stable when {@link TensorGenerator} runs, so caching is correctness-safe.
   */
  protected Set<List<Dimension<?>>> getShapes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    return getShapes(builder, node, valueNumber, false);
  }

  /**
   * Mode-aware variant of {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)}
   * (wala/ML#716): in exact mode, any points-to member whose shapes do not resolve poisons the
   * result to ⊤, so the returned set is the complete set of possibilities; in the default mode,
   * unresolvable members drop and the result is the resolvable subset.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code valueNumber}.
   * @param valueNumber The value number to resolve.
   * @param exact Whether an unresolvable union member poisons the result to ⊤.
   * @return The possible shapes, {@code null} (⊤), or an empty set (⊥).
   */
  protected Set<List<Dimension<?>>> getShapes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber, boolean exact) {
    ShapeResult result = this.getShapeResult(builder, node, valueNumber, exact);
    // The default mode's contract has always been "the resolvable subset" (wala/ML#716), and a
    // partial's members are exactly that subset, so they stand; only an exact read collapses a
    // partial to ⊤ at this legacy view (wala/ML#718).
    if (!exact && result.isPartial()) return result.members();
    return result.toLegacy();
  }

  /**
   * Record-carrying core of {@link #getShapes(PropagationCallGraphBuilder, CGNode, int, boolean)}
   * (wala/ML#718): the {@link ShapeResult} keeps an unresolvable remainder explicit alongside the
   * resolvable members instead of collapsing the whole result to ⊤.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code valueNumber}.
   * @param valueNumber The value number to resolve.
   * @param exact Whether an unresolvable union member poisons the result rather than dropping.
   * @return The resolution result.
   */
  protected ShapeResult getShapeResult(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber, boolean exact) {
    ShapeQuery key = new ShapeQuery(node, valueNumber, exact);
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    // See memoizedShapeResult on the null-engine (outside-an-analysis) fallback.
    if (engine == null) return computeShapes(builder, node, valueNumber, exact);
    java.util.function.Supplier<Object> transfer =
        () -> computeShapes(builder, node, valueNumber, exact);
    return (ShapeResult)
        (engine.isEvaluating()
            ? engine.read(key, transfer, true)
            : engine.demand(key, transfer, true));
  }

  private ShapeResult computeShapes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber, boolean exact) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    LOGGER.fine(
        () ->
            "getShapes(node, vn): node="
                + describe(node)
                + ", vn="
                + valueNumber
                + ", ptsEmpty="
                + valuePointsToSet.isEmpty());

    ShapeResult partial = null;
    if (!valuePointsToSet.isEmpty()) {
      ShapeResult fromValue = this.getShapeResultOfValue(builder, valuePointsToSet, exact);
      if (!fromValue.isBottom()) {
        if (!fromValue.hasUnknown()) return fromValue;
        // An unresolvable (⊤) result for a Keras call input can still be seeded from an explicit
        // `model.build(input_shape=...)` contract: the runtime value exists (non-empty PTS) but
        // its shape does not resolve, exactly the case the declared contract covers
        // (wala/ML#717).
        if (fromValue.members().isEmpty()) {
          Set<List<Dimension<?>>> contract =
              this.contractSeedForCallInput(builder, node, valueNumber);
          if (contract != null && !contract.isEmpty()) return ShapeResult.of(contract);
        }
        // An unknown remainder commonly reflects context collapse in a synthetic parameter, not
        // genuine runtime possibilities, so fall through to the per-context producer and
        // caller-walk paths, which are themselves exact; when those also fail, the partial's
        // resolvable members still stand alongside the remainder (wala/ML#716, wala/ML#718).
        partial = fromValue;
      }
    }

    // points-to set is empty. Try to find a generator for this variable.
    boolean implicit = builder.getPropagationSystem().isImplicit(valuePK);
    PointsToSetVariable var = null;
    if (!implicit) {
      var = builder.getPropagationSystem().findOrCreatePointsToSet(valuePK);
    }

    if (var != null) {
      try {
        TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
        // Recurse into the generator as long as it's not the *same* generator (identical source)
        // as `this`. Previously this check compared classes, which prevented an
        // `ElementWiseOperation` from recursing into a nested `ElementWiseOperation` on a
        // different operand value number — exactly the case for `(x - k1) / k2` chains.
        if (generator != null && !generator.getClass().equals(this.getClass())) {
          LOGGER.fine(
              () ->
                  "getShapes(node, vn): recovering via factory generator "
                      + generator.getClass().getSimpleName()
                      + " for vn="
                      + valueNumber);
          ShapeResult generatorShapes = memoizedShapeResult(builder, generator);
          // A ⊤ producer result cannot improve on a saved partial's members (wala/ML#718).
          if (generatorShapes.members().isEmpty() && partial != null) return partial;
          return generatorShapes;
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "getShapes(node, vn): factory IAE for vn=" + valueNumber + ": " + e.getMessage(),
            e);
      }
    }

    // No direct generator. Try tracing the definition or parameters.
    SSAInstruction def = node.getDU().getDef(valueNumber);

    // A value defined by an invoke that neither the points-to set nor the factory resolved is
    // typically a Python-callable result whose allocation the PA lost through a synthetic return
    // (e.g., a chained layer-call result in a `call` body). The callee's own return value still
    // resolves per context — recursing through a trampoline body reaches the underlying op result
    // and its generator — so descend into each callee's returns (wala/ML#718).
    if (def instanceof SSAAbstractInvokeInstruction) {
      ShapeResult fromCallees =
          this.shapeResultOfCalleeReturnValues(
              builder, node, (SSAAbstractInvokeInstruction) def, exact);
      if (!fromCallees.members().isEmpty()) return fromCallees;
      if (fromCallees.hasUnknown() && partial == null) partial = ShapeResult.unknown();
    }

    if (def == null) {
      // It's a parameter. Trace back to call sites.
      int paramPos = -1;
      for (int i = 0; i < node.getIR().getNumberOfParameters(); i++) {
        if (node.getIR().getParameter(i) == valueNumber) {
          paramPos = node.getMethod().isStatic() ? i : i - 1;
          break;
        }
      }

      if (paramPos >= -1) { // -1 is 'self'
        boolean callerUnknown = false;
        Set<List<Dimension<?>>> combinedRet = HashSetFactory.make();
        for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
            getCallerInvokes(builder, node)) {
          CGNode caller = callerInvoke.fst;
          SSAAbstractInvokeInstruction call = callerInvoke.snd;
          int argVn = -1;
          if (paramPos == -1) { // self
            argVn = call.getUse(0);
          } else if (call instanceof PythonInvokeInstruction) {
            // Try to find the argument index. This is simplified.
            if (paramPos + 1 < call.getNumberOfUses()) {
              argVn = call.getUse(paramPos + 1);
            }
          } else if (paramPos < call.getNumberOfUses()) {
            argVn = call.getUse(paramPos);
          }

          if (argVn != -1) {
            ShapeResult argShapes = this.getShapeResult(builder, caller, argVn, exact);
            // In exact mode an unresolvable caller marks the union's unknown remainder; the
            // default mode keeps the resolvable subset (wala/ML#716, wala/ML#718).
            if (argShapes.hasUnknown() && exact) callerUnknown = true;
            combinedRet.addAll(argShapes.members());
          }
        }
        if (!combinedRet.isEmpty()) return new ShapeResult(combinedRet, callerUnknown);

        // The runtime arguments did not resolve. For a Keras `call`'s first input, an explicitly
        // invoked `model.build(input_shape=(...))` with a resolvable literal declares the input
        // contract (built layers validate call inputs against the built shape), so seed the
        // parameter from it (wala/ML#717).
        Set<List<Dimension<?>>> contract =
            this.contractSeedForCallInput(builder, node, valueNumber);
        if (contract != null && !contract.isEmpty()) return ShapeResult.of(contract);
        if (callerUnknown) partial = ShapeResult.unknown();
      }
    }

    // The value is untraceable (empty points-to set and no recoverable def/parameter chain). Floor
    // to ⊥ (not a tensor) rather than throwing: this preserves the behavior from when the throw was
    // caught upstream and treated as not-a-tensor, but removes the abort risk for any caller that
    // doesn't catch it. wala/ML#620, mirroring the non-aborting floors in wala/ML#604 and
    // wala/ML#611.
    // The exact-mode union stayed incomplete: the partial's resolvable members stand alongside
    // the unknown remainder (wala/ML#716, wala/ML#718).
    if (partial != null) return partial;
    LOGGER.fine(
        () ->
            "Could not trace shape for value number "
                + valueNumber
                + " in "
                + describe(node)
                + "; flooring to ⊥ (not a tensor). wala/ML#620.");
    return ShapeResult.bottom();
  }

  /**
   * PTS-first wrapper: tries {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)} and falls
   * back to the SSA-substrate DU walk in {@link #shapesFromSSAChain} if that throws {@link
   * IllegalArgumentException} (implicit PK with empty PTS).
   *
   * <p>Every recursive step re-enters this wrapper so non-implicit intermediates (e.g., mnist
   * receivers reachable via factory dispatch) shortcut the walk.
   *
   * <p>If {@code getShapes} threw IAE and the DU walk doesn't find a concrete shape either, the IAE
   * is rethrown so callers relying on it to fail identification aren't misled.
   *
   * @param builder The propagation call graph builder.
   * @param node The CG node whose IR contains {@code vn}.
   * @param vn The SSA value number to resolve.
   * @return The resolved shapes (non-empty) from either path, or {@code null} / the empty set if
   *     {@code getShapes} returned those and the DU walk didn't help either. Rethrows IAE when
   *     {@code getShapes} throws and the DU walk doesn't help.
   */
  protected Set<List<Dimension<?>>> getShapesOrSSAChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    try {
      Set<List<Dimension<?>>> shapes = getShapes(builder, node, vn);
      if (shapes != null && !shapes.isEmpty()) return shapes;
      // `getShapes` returned null/empty without throwing — either "tensor with unknown shape"
      // (null) or "not a tensor" (empty). Try the DU walk for a concrete shape; fall through
      // to the original result if the walk doesn't help.
      Set<List<Dimension<?>>> chain = shapesFromSSAChain(builder, node, vn);
      if (chain != null && !chain.isEmpty()) return chain;
      return shapes;
    } catch (IllegalArgumentException e) {
      // IAE means "empty PTS and couldn't trace properties" — the caller is relying on this
      // to fail identification. Only override it if the SSA chain actually recovers a
      // concrete shape; otherwise rethrow so callers aren't misled into identifying
      // non-tensor parameters as tensors.
      Set<List<Dimension<?>>> chain = shapesFromSSAChain(builder, node, vn);
      if (chain != null && !chain.isEmpty()) return chain;
      throw e;
    }
  }

  /**
   * SSA-substrate shape lookup: walks the DU chain from {@code vn} backward, peeling tuple-unpack
   * {@link PythonPropertyRead}s and stopping at invokes it recognises.
   *
   * <p>Handles: mnist x_train/y_train/x_test/y_test invokes (hardcoded shapes via {@link
   * MnistInputData}); astype invokes (shape-preserving, recurses on the astype receiver);
   * tuple-field reads (peel via {@link #findTupleFieldStore}). Returns {@code null} for any other
   * creator kind — preserves today's ⊤-fallback rather than guessing.
   *
   * <p>Complements {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)}'s assignment-graph
   * walk: that one walks PTS propagation edges; this one walks SSA def-use edges. Where PTS is
   * implicit (summary-method returns), the PTS walk crashes on {@code findOrCreatePointsToSet} and
   * the DU walk is the only viable path.
   *
   * <p>TODO(wala/ML#402): delete once wala/WALA#1889 lands and summary-method returns materialise
   * concrete PTS — the normal PTS path will then recover these shapes via factory recursion.
   *
   * @param builder The propagation call graph builder used to resolve callees.
   * @param node The CG node whose IR contains {@code vn}.
   * @param vn The SSA value number whose shape we need.
   * @return The resolved shapes, or {@code null} if the DU walk doesn't recognise the creator.
   */
  protected Set<List<Dimension<?>>> shapesFromSSAChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    return shapesFromSSAChain(builder, node, vn, HashSetFactory.make());
  }

  /**
   * Recursive worker for {@link #shapesFromSSAChain(PropagationCallGraphBuilder, CGNode, int)}.
   *
   * @param builder The propagation call graph builder used to resolve callees.
   * @param node The CG node whose IR contains {@code vn}.
   * @param vn The SSA value number whose shape we need.
   * @param visited The set of value numbers already visited on this walk; used to break cycles when
   *     the DU chain loops through phi nodes or self-referential definitions.
   * @return The resolved shapes, or {@code null} if the DU walk doesn't recognise the creator or
   *     has already visited {@code vn} on this walk (cycle guard).
   */
  private Set<List<Dimension<?>>> shapesFromSSAChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn, Set<Integer> visited) {
    if (!visited.add(vn)) return null; // cycle guard
    SSAInstruction def = node.getDU().getDef(vn);
    LOGGER.fine(
        () ->
            "shapesFromSSAChain: entered, vn="
                + vn
                + ", def="
                + (def == null ? "null" : def.getClass().getSimpleName()));

    // Peel tuple-unpack: `x, y = a, b` lowers to `tmp = Tuple(a, b); x = tmp[0]; y = tmp[1]`.
    // Find the store that wrote the matching field on the same tuple and trace its stored vn.
    if (def instanceof PythonPropertyRead) {
      PythonPropertyRead propRead = (PythonPropertyRead) def;
      int storedVn = findTupleFieldStore(node, propRead);
      if (storedVn > 0) {
        LOGGER.fine(
            () ->
                "shapesFromSSAChain: peeled PythonPropertyRead at vn="
                    + vn
                    + " to storedVn="
                    + storedVn);
        try {
          Set<List<Dimension<?>>> viaPts = getShapes(builder, node, storedVn);
          if (viaPts != null && !viaPts.isEmpty()) return viaPts;
        } catch (IllegalArgumentException e) {
          // fall through
        }
        return shapesFromSSAChain(builder, node, storedVn, visited);
      }
      return null;
    }

    if (!(def instanceof SSAAbstractInvokeInstruction)) return null;
    SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) def;
    for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, call.getCallSite())) {
      TypeReference declaring = callee.getMethod().getReference().getDeclaringClass();
      if (declaring.equals(TensorFlowTypes.MNIST_X_TRAIN))
        return Set.of(MnistInputData.X_TRAIN_SHAPE);
      if (declaring.equals(TensorFlowTypes.MNIST_Y_TRAIN))
        return Set.of(MnistInputData.Y_TRAIN_SHAPE);
      if (declaring.equals(TensorFlowTypes.MNIST_X_TEST))
        return Set.of(MnistInputData.X_TEST_SHAPE);
      if (declaring.equals(TensorFlowTypes.MNIST_Y_TEST))
        return Set.of(MnistInputData.Y_TEST_SHAPE);
      if (declaring.equals(TensorFlowTypes.CIFAR10_X_TRAIN))
        return Set.of(Cifar10InputData.X_TRAIN_SHAPE);
      if (declaring.equals(TensorFlowTypes.CIFAR10_Y_TRAIN))
        return Set.of(Cifar10InputData.Y_TRAIN_SHAPE);
      if (declaring.equals(TensorFlowTypes.CIFAR10_X_TEST))
        return Set.of(Cifar10InputData.X_TEST_SHAPE);
      if (declaring.equals(TensorFlowTypes.CIFAR10_Y_TEST))
        return Set.of(Cifar10InputData.Y_TEST_SHAPE);
      if (declaring.equals(NumpyTypes.ASTYPE.getDeclaringClass())) {
        // astype preserves shape; recurse on its receiver.
        int astypeReceiverVn = propertyReadObjectRef(node, call);
        if (astypeReceiverVn > 0) {
          try {
            Set<List<Dimension<?>>> viaPts = getShapes(builder, node, astypeReceiverVn);
            if (viaPts != null && !viaPts.isEmpty()) return viaPts;
          } catch (IllegalArgumentException e) {
            // fall through
          }
          return shapesFromSSAChain(builder, node, astypeReceiverVn, visited);
        }
      }
      // Nested reshape is intentionally not handled here — the outer NdarrayReshape, if
      // present, computes its own shape via `getShapes`. In practice the chains we care
      // about ({reshape → EWO → from_tensor_slices}) route through EWO's binop handling,
      // which in turn re-enters this walk on its operand.
    }
    return null;
  }

  /**
   * Dtype counterpart to {@link #getShapesOrSSAChain(PropagationCallGraphBuilder, CGNode, int)}.
   * PTS-first via {@link #getDTypes(PropagationCallGraphBuilder, CGNode, int)}, SSA-DU fallback via
   * {@link #dtypesFromSSAChain}. Rethrows {@link IllegalArgumentException} from {@code getDTypes}
   * when the DU walk doesn't help, so callers that rely on IAE to fail identification are not
   * misled.
   *
   * @param builder The propagation call graph builder.
   * @param node The CG node whose IR contains {@code vn}.
   * @param vn The SSA value number whose dtype we need.
   * @return The resolved dtypes, or {@code null} if neither path finds a concrete dtype and {@code
   *     getDTypes} didn't throw.
   */
  protected Set<DType> getDTypesOrSSAChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    try {
      Set<DType> dtypes = getDTypes(builder, node, vn);
      // A ⊤-only result (just `UNKNOWN`) is no better than "nothing concrete": the SSA chain may
      // recover a real dtype through dtype-preserving ops (e.g. `reshape(pad(x))`), so prefer it
      // over ⊤. wala/ML#602. (A ⊤-only result previously short-circuited the chain; that was masked
      // until the wala/ML#603 catalog filter stopped a non-integer key from throwing here, which
      // had
      // incidentally produced an empty result that fell through to the chain.)
      boolean concrete =
          dtypes != null && !dtypes.isEmpty() && !(dtypes.size() == 1 && dtypes.contains(UNKNOWN));
      if (concrete) return dtypes;
      Set<DType> chain = dtypesFromSSAChain(builder, node, vn);
      if (chain != null && !chain.isEmpty()) return chain;
      return dtypes;
    } catch (IllegalArgumentException e) {
      Set<DType> chain = dtypesFromSSAChain(builder, node, vn);
      if (chain != null && !chain.isEmpty()) return chain;
      throw e;
    }
  }

  /**
   * Names of dtype-preserving TensorFlow ops: each returns a tensor with the same dtype as its
   * first tensor operand. Recognized syntactically by {@link #dtypesFromSSAChain} so chains through
   * them recover the underlying dtype, even when an op is unmodeled. See wala/ML#602.
   */
  private static final Set<String> DTYPE_PRESERVING_OP_NAMES =
      Set.of("reshape", "pad", "expand_dims", "squeeze", "transpose", "identity");

  /**
   * Dtype counterpart to {@link #shapesFromSSAChain(PropagationCallGraphBuilder, CGNode, int)}.
   * Handles: mnist invokes (all uint8); astype invokes (use astype's dtype arg — FLOAT32 default if
   * we can't resolve); tuple-unpack reads (peel via {@link #findTupleFieldStore}); binops involving
   * a float scalar literal (FLOAT32 promotion).
   *
   * <p>Returns {@code null} if the DU walk doesn't recognise the creator.
   */
  protected Set<DType> dtypesFromSSAChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    return dtypesFromSSAChain(builder, node, vn, HashSetFactory.make());
  }

  /**
   * Recursive worker for {@link #dtypesFromSSAChain(PropagationCallGraphBuilder, CGNode, int)}.
   *
   * @param builder The propagation call graph builder.
   * @param node The CG node whose IR contains {@code vn}.
   * @param vn The SSA value number whose dtype we need.
   * @param visited The set of value numbers already visited on this walk; used to break cycles.
   * @return The resolved dtypes, or {@code null} if the DU walk doesn't recognise the creator or
   *     has already visited {@code vn} on this walk (cycle guard).
   */
  private Set<DType> dtypesFromSSAChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn, Set<Integer> visited) {
    if (!visited.add(vn)) return null;
    SSAInstruction def = node.getDU().getDef(vn);

    if (def instanceof PythonPropertyRead) {
      PythonPropertyRead propRead = (PythonPropertyRead) def;
      int storedVn = findTupleFieldStore(node, propRead);
      if (storedVn > 0) {
        try {
          Set<DType> viaPts = getDTypes(builder, node, storedVn);
          if (viaPts != null && !viaPts.isEmpty()) return viaPts;
        } catch (IllegalArgumentException e) {
          // fall through
        }
        return dtypesFromSSAChain(builder, node, storedVn, visited);
      }
      return null;
    }

    if (def instanceof SSABinaryOpInstruction) {
      // Binop: check for float-literal promotion. If either operand is a Python Double/Float
      // constant, the result is FLOAT32 — same promotion rule as
      // `ElementWiseOperation.getDefaultDTypes` applies.
      int xVn = def.getUse(0);
      int yVn = def.getUse(1);
      if (isFloatLiteralVn(node, xVn) || isFloatLiteralVn(node, yVn)) {
        return EnumSet.of(DType.FLOAT32);
      }
      // Otherwise take the first operand's dtype.
      return dtypesFromSSAChain(builder, node, xVn, visited);
    }

    if (!(def instanceof SSAAbstractInvokeInstruction)) return null;
    SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) def;
    for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, call.getCallSite())) {
      TypeReference declaring = callee.getMethod().getReference().getDeclaringClass();
      if (declaring.equals(TensorFlowTypes.MNIST_X_TRAIN)
          || declaring.equals(TensorFlowTypes.MNIST_Y_TRAIN)
          || declaring.equals(TensorFlowTypes.MNIST_X_TEST)
          || declaring.equals(TensorFlowTypes.MNIST_Y_TEST)
          || declaring.equals(TensorFlowTypes.CIFAR10_X_TRAIN)
          || declaring.equals(TensorFlowTypes.CIFAR10_Y_TRAIN)
          || declaring.equals(TensorFlowTypes.CIFAR10_X_TEST)
          || declaring.equals(TensorFlowTypes.CIFAR10_Y_TEST)) {
        return EnumSet.of(DType.UINT8);
      }
      if (declaring.equals(NumpyTypes.ASTYPE.getDeclaringClass())) {
        // astype result takes its target dtype from the call's dtype arg. For now we return
        // FLOAT32, which is the astype-use we see in practice (`x.astype(np.float32)` in the
        // mnist chain) and also `AstypeOperation`'s fallback default. A precise lookup would
        // resolve the arg via `FIELD_REFERENCE_TO_DTYPE` — not yet implemented here.
        return EnumSet.of(DType.FLOAT32);
      }
    }

    // Dtype-preserving ops (`tf.reshape`, `tf.pad`, `tf.expand_dims`, ...) return a tensor with the
    // same dtype as their first tensor operand. Recognize them syntactically by the called
    // attribute
    // name (so this covers unmodeled ops too, e.g. `tf.pad`, which resolves to no call-graph
    // target)
    // and recurse on that operand. This lets chains like `reshape(pad(x))` recover `x`'s dtype
    // rather than landing at ⊤ when no single op in the chain is itself dtype-modeled. See
    // wala/ML#602.
    String calledName = calledFunctionName(node, call);
    if (calledName != null
        && DTYPE_PRESERVING_OP_NAMES.contains(calledName)
        && call.getNumberOfUses() >= 2) {
      int inputVn = call.getUse(1);
      LOGGER.fine(
          () ->
              "Recovering dtype through dtype-preserving op "
                  + calledName
                  + ": recursing from vn="
                  + vn
                  + " onto operand vn="
                  + inputVn
                  + ".");
      try {
        Set<DType> viaPts = getDTypes(builder, node, inputVn);
        if (viaPts != null && !viaPts.isEmpty()) return viaPts;
      } catch (IllegalArgumentException e) {
        // Fall through to the SSA-DU recursion.
      }
      return dtypesFromSSAChain(builder, node, inputVn, visited);
    }
    return null;
  }

  /**
   * Returns the called attribute's name for a function/method-style invoke {@code obj.name(...)} by
   * reading the member of the {@link PythonPropertyRead} that def'd the invoke's function object.
   * Used to recognize dtype-preserving ops (e.g. {@code tf.reshape}, {@code tf.pad}) by name,
   * including unmodeled ones that resolve to no call-graph target.
   *
   * @param node The {@link CGNode} whose IR contains {@code call}.
   * @param call The invoke whose called-attribute name is wanted.
   * @return The called attribute's name, or {@code null} if it can't be resolved to a string
   *     constant.
   */
  protected static String calledFunctionName(CGNode node, SSAAbstractInvokeInstruction call) {
    if (call.getNumberOfUses() < 1) return null;
    SSAInstruction funcDef = node.getDU().getDef(call.getUse(0));
    if (funcDef instanceof PythonPropertyRead) {
      int memberVn = ((PythonPropertyRead) funcDef).getMemberRef();
      SymbolTable st = node.getIR().getSymbolTable();
      if (st.isStringConstant(memberVn)) return st.getStringValue(memberVn);
    }
    return null;
  }

  /**
   * Returns true iff {@code vn} is a Python float-literal constant in {@code node}'s symbol table
   * &mdash; i.e., a {@link Double} or {@link Float} value with no defining instruction. Used for
   * scalar-literal dtype-promotion in binops ({@code int_tensor / 255.0} → float32).
   *
   * @param node The CG node whose symbol table to query.
   * @param vn The SSA value number to check.
   * @return {@code true} iff {@code vn} is a {@link Double} or {@link Float} literal constant.
   */
  protected static boolean isFloatLiteralVn(CGNode node, int vn) {
    if (vn <= 0) return false;
    if (node.getDU().getDef(vn) != null) return false;
    if (!node.getIR().getSymbolTable().isConstant(vn)) return false;
    Object val = node.getIR().getSymbolTable().getConstantValue(vn);
    return val instanceof Double || val instanceof Float;
  }

  /**
   * Scans {@code node}'s IR for a {@link PythonPropertyWrite} whose {@code objectRef} and member
   * value match {@code propRead}'s. Used to peel tuple-unpack patterns like {@code x, y = a, b}.
   *
   * @param node The CG node whose IR to scan.
   * @param propRead The read whose matching store we seek.
   * @return The stored value's SSA value number, or {@code -1} if no unique match is found.
   */
  protected static int findTupleFieldStore(CGNode node, PythonPropertyRead propRead) {
    int objectRef = propRead.getObjectRef();
    int memberRef = propRead.getMemberRef();
    int found = -1;
    for (SSAInstruction inst : node.getIR().getInstructions()) {
      if (!(inst instanceof PythonPropertyWrite)) continue;
      PythonPropertyWrite write = (PythonPropertyWrite) inst;
      if (write.getUse(0) != objectRef) continue;
      if (write.getUse(1) != memberRef) continue;
      int stored = write.getUse(2);
      if (found != -1 && found != stored) return -1; // ambiguous
      found = stored;
    }
    return found;
  }

  /**
   * For a method-style invoke {@code x.m(...)}, returns the receiver's SSA value number by reading
   * the {@code objectRef} of the {@link PythonPropertyRead} that def'd the invoke's function
   * object.
   *
   * @param node The CG node whose IR contains {@code call}.
   * @param call The invoke instruction to inspect.
   * @return The receiver's SSA value number, or {@code -1} if the invoke isn't a method-style call.
   */
  protected static int propertyReadObjectRef(CGNode node, SSAAbstractInvokeInstruction call) {
    if (call.getNumberOfUses() < 1) return -1;
    SSAInstruction funcDef = node.getDU().getDef(call.getUse(0));
    if (funcDef instanceof PythonPropertyRead) {
      return ((PythonPropertyRead) funcDef).getObjectRef();
    }
    return -1;
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valuePointsToSet The points-to set of the value from which the shape will be derived.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    return getShapesOfValue(builder, valuePointsToSet, false);
  }

  /**
   * Mode-aware variant of {@link #getShapesOfValue(PropagationCallGraphBuilder, OrdinalSet)}
   * (wala/ML#716): in exact mode, a member whose shapes do not resolve poisons the union to ⊤
   * instead of silently dropping.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valuePointsToSet The points-to set of the value from which the shape will be derived.
   * @param exact Whether an unresolvable union member poisons the result to ⊤.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapesOfValue(
      PropagationCallGraphBuilder builder,
      OrdinalSet<InstanceKey> valuePointsToSet,
      boolean exact) {
    return this.getShapeResultOfValue(builder, valuePointsToSet, exact).toLegacy();
  }

  /**
   * Record-carrying core of {@link #getShapesOfValue(PropagationCallGraphBuilder, OrdinalSet,
   * boolean)} (wala/ML#718): in exact mode, a member whose shapes do not resolve marks the unknown
   * remainder and the resolvable members keep collecting, so a partially resolvable points-to union
   * degrades per member instead of collapsing to ⊤. In the default mode unresolvable members still
   * drop silently.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valuePointsToSet The points-to set of the value from which the shape will be derived.
   * @param exact Whether an unresolvable union member marks the unknown remainder rather than
   *     dropping.
   * @return The resolution result.
   */
  protected ShapeResult getShapeResultOfValue(
      PropagationCallGraphBuilder builder,
      OrdinalSet<InstanceKey> valuePointsToSet,
      boolean exact) {
    if (valuePointsToSet == null || valuePointsToSet.isEmpty()) {
      LOGGER.fine(
          () ->
              "Empty points-to set for value in source: "
                  + this.getSource()
                  + ". Returning unknown (⊤).");
      return ShapeResult.unknown();
    }

    boolean hasUnknown = false;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet)
      if (valueIK instanceof ConstantKey) ret.add(emptyList()); // Scalar value.
      else if (valueIK instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        TypeReference reference = asin.concreteType().getReference();

        if (reference.equals(list) || reference.equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              pointerAnalysis.getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

          LOGGER.fine(
              "The object catalog points-to set size is: " + objectCatalogPointsToSet.size() + ".");

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
            Integer fieldIndex = getFieldIndex(constantKey);
            // Skip non-integer attribute keys (e.g. method-name fields); they aren't elements.
            // See wala/ML#603.
            if (fieldIndex == null) continue;

            FieldReference subscript =
                FieldReference.findOrCreate(
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

            IField f = builder.getClassHierarchy().resolveField(subscript);
            LOGGER.fine("Found field: " + f);

            PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
            LOGGER.fine(
                "Found pointer key for instance field: "
                    + describe(pointerKeyForInstanceField)
                    + ".");

            OrdinalSet<InstanceKey> instanceFieldPointsToSet =
                pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
            LOGGER.fine(
                "Points-to set for instance field: " + describe(instanceFieldPointsToSet) + ".");

            Set<List<Dimension<?>>> shapesOfField =
                this.getShapesOfValue(builder, instanceFieldPointsToSet, exact);

            if (shapesOfField == null) {
              // An unresolvable element makes the union incomplete: mark the remainder and keep
              // collecting the resolvable members (wala/ML#718).
              if (exact) hasUnknown = true;
              continue;
            }

            for (List<Dimension<?>> shapeList : shapesOfField) {
              List<Dimension<?>> shape = new ArrayList<>();

              // Leading dim is the element count, i.e. the integer-indexed catalog entries only;
              // the raw catalog size is inflated by non-integer attribute keys. See wala/ML#603.
              shape.add(new NumericDim(integerCatalogSize(objectCatalogPointsToSet)));
              shape.addAll(shapeList);

              ret.add(shape);
            }
          }
        } else if (reference.equals(TensorFlowTypes.D_TYPE)) {
          LOGGER.fine("Ignoring DType: " + describe(asin));
        } else if (reference.equals(TensorFlowTypes.FEATURE)) {
          LOGGER.fine("Ignoring feature: " + describe(asin));
        } else {
          // Assume the value is a tensor and attempt to find the generator that created it
          // to ask for its shape.
          LOGGER.fine(
              "Encountered "
                  + reference.getName()
                  + ". Attempting to retrieve shape from producer.");
          ShapeResult fromTensor = this.getShapeResultFromTensor(builder, asin, exact);
          LOGGER.fine(
              () ->
                  "DIAG creator [alloc="
                      + asin.getNode().getMethod().getDeclaringClass().getName()
                      + "@"
                      + asin.getSite().getProgramCounter()
                      + "] => members="
                      + fromTensor.members().size()
                      + " unk="
                      + fromTensor.hasUnknown());
          // An empty result means no producer resolved for this member, not a scalar.
          if (fromTensor.hasUnknown() || fromTensor.isBottom()) {
            if (exact) hasUnknown = true; // The union is incomplete, wala/ML#718.
          }
          ret.addAll(fromTensor.members());
        }
      } else if (getAllocationSiteInNode(valueIK) != null) {
        // Unwrap ScopeMappingInstanceKey or similar wrapping keys
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);

        // Instead of forcing a points-to set, try to get the generator for this allocation site
        PointerKey pk =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(asin.getNode(), asin.getSite().getProgramCounter());
        PointsToSetVariable var = null;
        if (!builder.getPropagationSystem().isImplicit(pk)) {
          var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
        }
        try {
          TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
          if (generator != null && !generator.getClass().equals(this.getClass())) {
            ShapeResult generatorShapes = memoizedShapeResult(builder, generator);
            ret.addAll(generatorShapes.members());
            if (exact && generatorShapes.hasUnknown())
              hasUnknown = true; // Incomplete union, wala/ML#718.
          } else if (exact) hasUnknown = true;
        } catch (IllegalArgumentException e) {
          // Not a recognized generator.
          LOGGER.log(Level.FINE, "No generator found for variable: " + var, e);
          if (exact) hasUnknown = true;
        }
      } else {
        throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");
      }

    return new ShapeResult(ret, hasUnknown);
  }

  /**
   * Retrieves the shapes of a tensor that is the result of another TensorFlow operation (the
   * "producer").
   *
   * <p>This method traces the tensor back to its allocation site to identify the operation that
   * created it. It handles two main scenarios:
   *
   * <ol>
   *   <li><b>Direct Allocation in `do`:</b> The tensor is allocated directly within the `do` method
   *       of the operation. It attempts to find the definition of the tensor and trace it back to a
   *       {@link PointsToSetVariable}. If successful, it delegates to the generator for that
   *       source. If points-to analysis fails (e.g., due to implicit pointer keys), it attempts to
   *       create a manual generator using {@link #createManualGenerator(CGNode,
   *       PropagationCallGraphBuilder)}.
   *   <li><b>Helper Method (`read_data`):</b> The tensor is allocated in a helper method (like
   *       `read_data`) called by `do`. It identifies the call site in `do` that invoked the helper
   *       and recursively delegates to the generator for the result of that call.
   * </ol>
   *
   * @param builder the propagation call graph builder
   * @param asin the allocation site of the tensor
   * @return the resolution result; ⊥ when no producer resolves
   */
  private ShapeResult getShapeResultFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin, boolean exact) {
    // A producer chain that re-encounters an allocation site is a cycle in the value's producer
    // graph (a loop-carried tensor whose points-to union includes its own downstream producers).
    // The engine converges the cycle from its non-cyclic base instead of recursing without bound
    // or inheriting a whole-read ⊤ (wala/ML#718).
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    // See memoizedShapeResult on the null-engine (outside-an-analysis) fallback.
    if (engine == null) return this.doGetShapeResultFromTensor(builder, asin, exact);
    Object key = Pair.make("shape-delegation", asin);
    java.util.function.Supplier<Object> transfer =
        () -> this.doGetShapeResultFromTensor(builder, asin, exact);
    return (ShapeResult)
        (engine.isEvaluating()
            ? engine.read(key, transfer, true)
            : engine.demand(key, transfer, true));
  }

  /**
   * Body of {@link #getShapeResultFromTensor(PropagationCallGraphBuilder, AllocationSiteInNode,
   * boolean)}, running as its engine query's transfer.
   *
   * @param builder the propagation call graph builder
   * @param asin the allocation site of the tensor
   * @param exact whether an unresolvable member marks the unknown remainder
   * @return the resolution result; ⊥ when no producer resolves
   */
  private ShapeResult doGetShapeResultFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin, boolean exact) {
    boolean hasUnknown = false;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    CGNode readDataNode = asin.getNode();

    // Support allocations directly in 'do' methods (preferred for 1-CFA context separation).
    if (readDataNode.getMethod().getName().toString().equals(DO_METHOD_NAME)) {
      int def = findDefinition(readDataNode, asin);
      if (def != -1) {
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(readDataNode, def);
        PointsToSetVariable defSource = null;
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }

        TensorGenerator generator =
            createManualGenerator(readDataNode, asin.concreteType().getReference(), builder);

        if (generator != null) {
          // Avoid infinite recursion for manual generators
          if (this.manualNode != null && this.manualNode.equals(readDataNode)) {
            return ShapeResult.bottom();
          }
          LOGGER.fine("Delegating shape inference to: " + generator);
          ShapeResult delegatedShapes = memoizedShapeResult(builder, generator);
          ret.addAll(delegatedShapes.members());
          if (exact && delegatedShapes.hasUnknown())
            hasUnknown = true; // Incomplete union, wala/ML#718.
        } else if (defSource != null) {
          // Avoid infinite recursion if the current generator is for the same source.
          if (this.getSource() != null && this.getSource().equals(defSource)) {
            return ShapeResult.bottom();
          }
          try {
            generator = TensorGeneratorFactory.getGenerator(defSource, builder);
          } catch (IllegalArgumentException e) {
            // Factory couldn't resolve — treat as "no generator" and skip. See wala/ML#363.
            LOGGER.log(Level.FINE, "Delegating shape inference: factory IAE for " + defSource, e);
            generator = null;
          }
          if (generator != null) {
            LOGGER.fine("Delegating shape inference to: " + generator);
            ShapeResult delegatedShapes = memoizedShapeResult(builder, generator);
            ret.addAll(delegatedShapes.members());
            if (exact && delegatedShapes.hasUnknown())
              hasUnknown = true; // Incomplete union, wala/ML#718.
          } else if (exact) hasUnknown = true;
        }
      }
      return new ShapeResult(ret, hasUnknown);
    }

    // 1. read_data is called by the operation's 'do' method; the call-graph edges identify the
    // calling instructions.
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, readDataNode)) {
      CGNode doNode = callerInvoke.fst;
      SSAAbstractInvokeInstruction call = callerInvoke.snd;

      // Construct a source for the result of this call (which is the tensor object).
      if (call.getNumberOfDefs() > 0) {
        int def = call.getDef();
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(doNode, def);
        PointsToSetVariable defSource = null;
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }

        // Try to create a manual generator for the caller (doNode) first.
        TensorGenerator generator = createManualGenerator(doNode, builder);
        if (generator == null) {
          try {
            generator = TensorGeneratorFactory.getGenerator(defSource, builder);
          } catch (IllegalArgumentException e) {
            // Factory couldn't resolve — treat as "no generator". See wala/ML#363.
            LOGGER.log(Level.FINE, "Delegating shape inference: factory IAE for " + defSource, e);
            generator = null;
          }
        }

        if (generator != null) {
          LOGGER.fine("Delegating shape inference to: " + generator);
          Set<List<Dimension<?>>> delegatedShapes = generator.getShapes(builder);
          if (delegatedShapes != null) ret.addAll(delegatedShapes);
          else if (exact) hasUnknown = true; // Incomplete union, wala/ML#718.
        } else if (exact) hasUnknown = true;
      }
    }
    return new ShapeResult(ret, hasUnknown);
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the dtype argument, which is expected to be a set of
   *     type literals.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected Set<DType> getDTypesFromDTypeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    if (pointsToSet == null || !pointsToSet.iterator().hasNext())
      throw new IllegalArgumentException(
          "Empty points-to set for dtype argument in source: " + describe(this.getSource()) + ".");

    Set<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      if (asin == null && !(instanceKey instanceof ConstantKey)) continue;
      // First, check for `None`.
      if (instanceKey instanceof ConstantKey) {
        ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
        Object value = constantKey.getValue();

        if (value == null) {
          LOGGER.fine(
              "DType argument is None for source: "
                  + this.getSource()
                  + "; using default dtypes."
                  + ".");
          return getDefaultDTypes(builder);
        }
      }

      // Check if it matches a known DType field (e.g., tf.float32 or np.float32). The owner
      // allocation may be a TENSORFLOW_TYPE (when the dtype came from `tf.float32`) or a
      // NUMPY_TYPE (when it came from `np.float32`); both are allocated in the same synthetic
      // import-method node that allocates the dtype itself.
      boolean found = false;
      if (instanceKey instanceof AllocationSiteInNode) {
        CGNode importNode = ((AllocationSiteInNode) instanceKey).getNode();

        if (importNode != null) {
          LOGGER.fine("Found import node of interest: " + describe(importNode) + ".");
          for (TypeReference ownerType :
              new TypeReference[] {TENSORFLOW_TYPE, TensorFlowTypes.NUMPY_TYPE}) {
            InstanceKey ownerIK =
                pointerAnalysis
                    .getHeapModel()
                    .getInstanceKeyForAllocation(importNode, NewSiteReference.make(0, ownerType));

            // Check dtype literals.
            for (Entry<FieldReference, DType> entry : FIELD_REFERENCE_TO_DTYPE.entrySet()) {
              FieldReference fieldRef = entry.getKey();
              DType dtype = entry.getValue();
              IField field = builder.getClassHierarchy().resolveField(fieldRef);

              if (field != null) {
                PointerKey pk =
                    pointerAnalysis.getHeapModel().getPointerKeyForInstanceField(ownerIK, field);

                OrdinalSet<InstanceKey> pts = pointerAnalysis.getPointsToSet(pk);
                if (pts != null) {
                  for (InstanceKey ik : pts)
                    if (ik.equals(instanceKey)) {
                      ret.add(dtype);
                      LOGGER.fine(
                          "Found dtype: "
                              + dtype
                              + " for source: "
                              + describe(this.getSource())
                              + " from dType: "
                              + describe(instanceKey)
                              + ".");
                      found = true;
                      break;
                    }
                }
              }
              if (found) break;
            }
            if (found) break;
          }
        }
      }

      if (found) continue;

      IClass concreteType = instanceKey.concreteType();
      TypeReference typeReference = concreteType.getReference();

      if (typeReference.equals(TensorFlowTypes.D_TYPE)) {
        // An unmodeled dtype: a `tf.DType` instance with no entry in `FIELD_REFERENCE_TO_DTYPE`
        // (e.g. a half-precision or quantized dtype not yet enumerated). Degrade to UNKNOWN (the ⊤
        // dtype) rather than throwing. When this resolves a parameter dtype during entrypoint
        // creation from an `input_signature`, throwing empties the entrypoint set and aborts the
        // whole call graph (wala/ML#637). Lose dtype precision for this one value rather than
        // killing the analysis, but log it so the modeling gap is still visible.
        LOGGER.warning(
            () ->
                "Unmodeled dtype: "
                    + describe(instanceKey)
                    + "; degrading to "
                    + DType.UNKNOWN
                    + ".");
        ret.add(DType.UNKNOWN);
      } else if (asin != null
          && asin.getNode()
              .getMethod()
              .getDeclaringClass()
              .getReference()
              .equals(CONSTANT.getDeclaringClass())) {
        // We have a `tf.constant(...)` result. Detect this by checking the
        // *containing method* of the allocation (the `tensorflow/functions/constant.do()`
        // CGNode), not the alloc's concrete type. The latter is the function-name
        // duplicate `Ltensorflow/python/framework/constant_op/constant`, which we want
        // to be able to migrate to canonical `Ltensorflow/python/framework/ops/Tensor`
        // later (wala/ML#459 PR 2). Reading the containing method's declaring class
        // instead decouples this recognition from the alloc-class convention.
        //
        // Use the already-unwrapped {@code asin} from the loop header (line 1299) so
        // wrapping {@link InstanceKey}s (e.g. {@link
        // com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKey}) route through this
        // branch correctly — the prior {@code instanceof AllocationSiteInNode} check
        // would have skipped wrapped keys (Copilot review on PR #216).
        //
        // Extract the alloc's `dtype` field and recurse to resolve the actual DType.
        IField valueField =
            builder.getClassHierarchy().resolveField(TensorFlowTypes.CONSTANT_DTYPE);
        PointerKey valuePK = builder.getPointerKeyForInstanceField(instanceKey, valueField);
        OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(valuePK);
        if (valuePts != null && !valuePts.isEmpty()) {
          ret.addAll(this.getDTypesFromDTypeArgument(builder, valuePts));
        }
      } else if (typeReference.equals(TensorFlowTypes.TENSOR_SPEC)
          || typeReference.equals(TensorFlowTypes.RAGGED_TENSOR_SPEC)) {
        // We have a TensorSpec or RaggedTensorSpec. We extract the 'dtype' field and recurse to
        // resolve the actual DType value (usually a tf.DType instance).
        IField dtypeField =
            builder
                .getClassHierarchy()
                .resolveField(
                    typeReference.equals(TensorFlowTypes.TENSOR_SPEC)
                        ? TensorFlowTypes.SPEC_DTYPE
                        : TensorFlowTypes.RAGGED_SPEC_DTYPE);
        PointerKey dtypePK = builder.getPointerKeyForInstanceField(instanceKey, dtypeField);
        OrdinalSet<InstanceKey> dtypePts = pointerAnalysis.getPointsToSet(dtypePK);
        if (dtypePts != null && !dtypePts.isEmpty()) {
          ret.addAll(this.getDTypesFromDTypeArgument(builder, dtypePts));
        }
      } else if (typeReference.equals(tuple) || typeReference.equals(list)) {
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Integer fieldIndex = getFieldIndex(constantKey);
          // Skip non-integer attribute keys (e.g. method-name fields); they aren't elements.
          // See wala/ML#603.
          if (fieldIndex == null) continue;

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f != null) {
            PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
            OrdinalSet<InstanceKey> fieldPts = pointerAnalysis.getPointsToSet(pk);
            ret.addAll(this.getDTypesFromDTypeArgument(builder, fieldPts));
          }
        }
      } else if (typeReference.equals(dict)) {
        // A dict-structured dtype specification, e.g. `output_types={"feature": tf.int32, ...}`.
        // `tf.data.Dataset.from_generator` accepts a nested structure of dtypes; the values (keyed
        // by arbitrary string names) are the per-leaf dtype specs. Recurse into each value and
        // union the results, mirroring the tuple/list handling above. See
        // https://github.com/wala/ML/issues/615.
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Object keyValue = constantKey.getValue();
          // Dict keys are strings; the value is stored as an instance field named by the key.
          if (!(keyValue instanceof String)) continue;

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom((String) keyValue), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f != null) {
            PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
            OrdinalSet<InstanceKey> fieldPts = pointerAnalysis.getPointsToSet(pk);
            ret.addAll(this.getDTypesFromDTypeArgument(builder, fieldPts));
          }
        }
      } else if (instanceKey instanceof ConstantKey
          && ((ConstantKey<?>) instanceKey).getValue() instanceof String) {
        String value = (String) ((ConstantKey<?>) instanceKey).getValue();
        DType dtype = null;

        try {
          dtype = DType.valueOf(value.toUpperCase()); // Validate the dtype string.
        } catch (IllegalArgumentException | NullPointerException e) {
          if (value.equals("float")) {
            dtype = FLOAT32;
          } else {
            throw new IllegalStateException("Unknown dtype string: " + value + ".", e);
          }
        }

        ret.add(dtype);
        LOGGER.fine(
            "Found dtype: "
                + dtype
                + " for source: "
                + this.getSource()
                + " from string: "
                + value
                + ".");
      } else {
        throw new IllegalStateException(
            "Expected a "
                + TensorFlowTypes.D_TYPE
                + " for the dtype, but got: "
                + typeReference
                + ".");
      }
    }

    return ret;
  }

  /**
   * Returns a set of possible dtypes of the tensor returned by this generator when an explicit
   * dtype isn't provided as an argument.
   *
   * <p>Implementations should return {@link DType#UNKNOWN} (i.e., {@code EnumSet.of(DType.UNKNOWN)}
   * or {@code Set.of(DType.UNKNOWN)}) to indicate that the dtype cannot be determined. An empty set
   * means the variable is not a tensor at all (⊥). A set of concrete dtypes means the dtype is
   * known.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible dtypes of the tensor returned by this generator when an explicit
   *     dtype isn't provided as an argument, or a set containing {@link DType#UNKNOWN} if the dtype
   *     cannot be determined.
   */
  protected abstract Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder);

  /**
   * Returns the libraries whose operations produce the value modeled by this generator
   * (wala/ML#724).
   *
   * <p>The default is {@link TensorOrigin#TENSORFLOW}: generators model TensorFlow APIs unless they
   * specifically model numpy ones ({@link NpArray}, {@link AstypeOperation}, etc.), which override
   * this to {@link TensorOrigin#NUMPY}. A generator implementing {@link DelegatingTensorGenerator}
   * answers with its underlying generator's origins, since the modeled value (a collection element,
   * an enumerated item) is produced by whatever produced the underlying container's contents.
   *
   * <p>Classification is by the runtime type of the produced value, not the namespace of the
   * invoked API; see {@link TensorOrigin}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The producing libraries of the modeled value.
   */
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    if (this instanceof DelegatingTensorGenerator) {
      TensorGenerator underlying = ((DelegatingTensorGenerator) this).getUnderlying();
      if (underlying != null && underlying != this) return underlying.getOrigins(builder);
    }
    return EnumSet.of(TensorOrigin.TENSORFLOW);
  }

  /**
   * Returns the value number for the dtype argument in the function call.
   *
   * @return The value number for the dtype argument in the function call or -1 if the dtype
   *     argument is not supported.
   */
  protected int getDTypeArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getDTypeParameterPosition());
  }

  protected abstract int getDTypeParameterPosition();

  protected abstract String getDTypeParameterName();

  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName(), true);
    if (valNum <= 0) return this.getDefaultDTypes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());

    // If the argument dtype is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultDTypes(builder);
    else
      // The dtype points-to set is non-empty, meaning that the dtype was explicitly set.
      return this.getDTypesFromDTypeArgument(builder, pointsToSet);
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator. The dtype is inferred
   * from the argument represented by the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valueNumber The value number of the argument from which to infer the dtype.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder, int valueNumber) {
    return getDTypes(builder, this.getNode(), valueNumber);
  }

  /**
   * Returns the possible dtypes of the tensor represented by the given value number in the
   * specified node. Memoized on {@code (node, valueNumber)} per builder by the worklist engine
   * (wala/ML#365); see {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)} for the
   * rationale.
   */
  protected Set<DType> getDTypes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    Pair<CGNode, Integer> key = Pair.make(node, valueNumber);
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    // See memoizedShapeResult on the null-engine (outside-an-analysis) fallback.
    if (engine == null) return computeDTypes(builder, node, valueNumber);
    java.util.function.Supplier<Object> transfer = () -> computeDTypes(builder, node, valueNumber);
    @SuppressWarnings("unchecked")
    Set<DType> diverted =
        (Set<DType>)
            (engine.isEvaluating()
                ? engine.read(key, transfer, false)
                : engine.demand(key, transfer, false));
    return diverted;
  }

  private Set<DType> computeDTypes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    // wala/ML#570: a property read of a NamedTuple/object field (e.g. `inputs.node_embeddings`)
    // often has no usable points-to set at the read site when the object was constructed in a
    // caller and threaded in as a parameter. Resolve its dtype by reading the named field off the
    // object's instance in the heap, where the positional-field write (wala/ML#579) is
    // materialized. Only a concrete result is returned, so this strictly adds precision over the
    // existing paths below.
    SSAInstruction propDef = node.getDU().getDef(valueNumber);
    if (propDef instanceof PythonPropertyRead) {
      Set<DType> viaField =
          this.dtypesFromInstanceField(builder, node, (PythonPropertyRead) propDef);
      if (viaField != null && !viaField.isEmpty() && !viaField.equals(EnumSet.of(DType.UNKNOWN)))
        return viaField;
    }

    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    if (valuePointsToSet != null && !valuePointsToSet.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, valuePointsToSet);
      if (!dtypes.isEmpty()) {
        return dtypes;
      }
    }

    // points-to set is empty or yielded no dtypes. Try to find a generator for this variable.
    PointsToSetVariable var = null;
    if (!builder.getPropagationSystem().isImplicit(valuePK)) {
      var = builder.getPropagationSystem().findOrCreatePointsToSet(valuePK);
    }

    if (var != null) {
      try {
        TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
        // See `getShapes(builder, CGNode, int)` — we compare by source, not class, so two
        // different `ElementWiseOperation` generators for different operand value numbers can
        // still recurse into each other.
        if (generator != null && !generator.getClass().equals(this.getClass())) {
          return memoizedDTypes(builder, generator);
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(Level.FINE, "Not a recognized generator: " + var, e);
      }
    }

    // No direct generator. Try tracing the definition or parameters.
    SSAInstruction def = node.getDU().getDef(valueNumber);
    if (def == null) {
      // It's a parameter. Trace back to call sites.
      int paramPos = -1;
      for (int i = 0; i < node.getIR().getNumberOfParameters(); i++) {
        if (node.getIR().getParameter(i) == valueNumber) {
          paramPos = node.getMethod().isStatic() ? i : i - 1;
          break;
        }
      }

      if (paramPos >= -1) {
        Set<DType> combinedRet = EnumSet.noneOf(DType.class);
        for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
            getCallerInvokes(builder, node)) {
          CGNode caller = callerInvoke.fst;
          SSAAbstractInvokeInstruction call = callerInvoke.snd;
          int argVn = -1;
          if (paramPos == -1) { // self
            argVn = call.getUse(0);
          } else if (call instanceof PythonInvokeInstruction) {
            if (paramPos + 1 < call.getNumberOfUses()) {
              argVn = call.getUse(paramPos + 1);
            }
          } else if (paramPos < call.getNumberOfUses()) {
            argVn = call.getUse(paramPos);
          }

          if (argVn != -1) {
            combinedRet.addAll(this.getDTypes(builder, caller, argVn));
          }
        }
        if (!combinedRet.isEmpty()) return combinedRet;
      }
    }

    // The value is untraceable (empty points-to set and no recoverable def/parameter chain). Floor
    // to ⊥ (not a tensor), paired with the shape floor above, rather than throwing: this preserves
    // the behavior from when the throw was caught upstream and treated as not-a-tensor, but removes
    // the abort risk for any caller that doesn't catch it. wala/ML#620.
    LOGGER.fine(
        () ->
            "Could not trace dtype for value number "
                + valueNumber
                + " in "
                + describe(node)
                + "; flooring to ⊥ (not a tensor). wala/ML#620.");
    return Collections.emptySet();
  }

  /**
   * Resolves the dtypes of a property read ({@code obj.field}) by reading the named field off
   * {@code obj}'s instance(s) in the heap, rather than from the read result's (often empty)
   * points-to set. This recovers the type of a field whose value was written in a different frame
   * &mdash; e.g. a {@code NamedTuple} constructed in a caller and threaded in as a parameter, whose
   * positional fields are materialized by the synthesized constructor (wala/ML#579). Generalizes
   * the receiver-field read that generators like {@link DenseCall} already use. See <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>.
   *
   * @param builder The propagation call graph builder.
   * @param node The CG node whose IR contains the property read.
   * @param propRead The property-read instruction.
   * @return The dtypes read from the field, or {@code null} if the field name, field, or any
   *     instance field could not be resolved.
   */
  protected Set<DType> dtypesFromInstanceField(
      PropagationCallGraphBuilder builder, CGNode node, PythonPropertyRead propRead) {
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    String fieldName = null;
    for (InstanceKey ik :
        pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, propRead.getMemberRef()))) {
      if (ik instanceof ConstantKey && ((ConstantKey<?>) ik).getValue() instanceof String) {
        fieldName = (String) ((ConstantKey<?>) ik).getValue();
        break;
      }
    }
    if (fieldName == null) return null;
    IField f =
        builder
            .getClassHierarchy()
            .resolveField(
                FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldName), Root));
    if (f == null) return null;
    Set<DType> ret = null;
    for (InstanceKey objIK :
        pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, propRead.getObjectRef()))) {
      AllocationSiteInNode asin = getAllocationSiteInNode(objIK);
      if (asin == null) continue;
      OrdinalSet<InstanceKey> fieldPTS =
          pa.getPointsToSet(builder.getPointerKeyForInstanceField(asin, f));
      Set<DType> d = this.getDTypesOfValue(builder, fieldPTS);
      if (d != null && !d.isEmpty()) {
        if (ret == null) ret = EnumSet.noneOf(DType.class);
        ret.addAll(d);
      }
    }
    return ret;
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator. The dtype is inferred
   * from the given points-to set.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valuePointsToSet The points-to set of the value from which the dtype will be derived.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected Set<DType> getDTypesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    if (valuePointsToSet == null || valuePointsToSet.isEmpty()) {
      LOGGER.fine(
          () ->
              "Empty points-to set for value in source: "
                  + this.getSource()
                  + ". Returning null (unknown).");
      return null;
    }

    Set<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet) {
      if (valueIK instanceof ConstantKey) { // It's a scalar value.
        ConstantKey<?> constantKey = (ConstantKey<?>) valueIK;
        Object value = constantKey.getValue();
        if (value instanceof Float || value instanceof Double) {
          ret.add(FLOAT32);
          LOGGER.fine(
              "Inferred dtype: "
                  + FLOAT32
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof Integer || value instanceof Long) {
          ret.add(INT32);
          LOGGER.fine(
              "Inferred dtype: "
                  + INT32
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof String) {
          ret.add(STRING);
          LOGGER.fine(
              "Inferred dtype: "
                  + STRING
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof Boolean) {
          ret.add(BOOL);
          LOGGER.fine(
              "Inferred dtype: "
                  + BOOL
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value != null) {
          // Unrecognized value type. Add UNKNOWN to `ret` and let the
          // end-of-function lattice-collapse normalize: if any path produces
          // UNKNOWN, the final result is {UNKNOWN} regardless of what other
          // iterations contributed. Same shape as the prior Boolean-case fix
          // (#447): missing types should fall through to ⊤ in the lattice
          // rather than terminate the analysis.
          ret.add(UNKNOWN);
          LOGGER.fine(
              "Unrecognized constant type for source: "
                  + this.getSource()
                  + " value: "
                  + value
                  + " ("
                  + value.getClass()
                  + "); contributing UNKNOWN dtype.");
        }
      } else if (valueIK instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        TypeReference reference = asin.concreteType().getReference();

        if (reference.equals(list) || reference.equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              pointerAnalysis.getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

          LOGGER.fine(
              "The object catalog points-to set size is: " + objectCatalogPointsToSet.size() + ".");

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
            Integer fieldIndex = getFieldIndex(constantKey);
            // Skip non-integer attribute keys (e.g. method-name fields); they aren't elements.
            // See wala/ML#603.
            if (fieldIndex == null) continue;

            FieldReference subscript =
                FieldReference.findOrCreate(
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

            IField f = builder.getClassHierarchy().resolveField(subscript);
            LOGGER.fine("Found field: " + f);

            PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
            LOGGER.fine(
                "Found pointer key for instance field: "
                    + describe(pointerKeyForInstanceField)
                    + ".");

            OrdinalSet<InstanceKey> instanceFieldPointsToSet =
                pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
            LOGGER.fine(
                "Points-to set for instance field: " + describe(instanceFieldPointsToSet) + ".");

            Set<DType> fieldDTypes = this.getDTypesOfValue(builder, instanceFieldPointsToSet);
            if (fieldDTypes != null) ret.addAll(fieldDTypes);
          }
        } else if (reference.equals(TensorFlowTypes.FEATURE)) {
          // Ignore features.
          LOGGER.fine("Ignoring feature: " + describe(asin));
        } else if (reference.equals(TensorFlowTypes.D_TYPE)) {
          // Ignore DTypes.
          LOGGER.fine("Ignoring DType: " + describe(asin));
        } else {
          // Assume the value is a tensor and attempt to find the generator that created it
          // to ask for its dtype.
          LOGGER.fine(
              "Encountered "
                  + reference.getName()
                  + ". Attempting to retrieve dtype from producer.");
          ret.addAll(this.getDTypesFromTensor(builder, asin));
        }
      } else if (getAllocationSiteInNode(valueIK) != null) {
        // Unwrap ScopeMappingInstanceKey or similar wrapping keys
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);

        // Instead of forcing a points-to set, try to get the generator for this allocation site
        PointerKey pk =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(asin.getNode(), asin.getSite().getProgramCounter());
        PointsToSetVariable var = null;
        if (!builder.getPropagationSystem().isImplicit(pk)) {
          var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
        }
        try {
          TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
          if (generator != null && !generator.getClass().equals(this.getClass())) {
            ret.addAll(memoizedDTypes(builder, generator));
          }
        } catch (IllegalArgumentException e) {
          // Factory couldn't resolve — skip this instance. See wala/ML#363.
          LOGGER.log(Level.FINE, "getDTypesOfValue: factory IAE for " + var, e);
        }
      } else {
        throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");
      }
    }

    // Normalize the lattice: ⊤ subsumes any concrete dtype, so if any
    // contributing path produced UNKNOWN (either directly via the
    // unrecognized-constant short-circuit above, or via a recursive call
    // for list/tuple element dtypes that propagated UNKNOWN through
    // `ret.addAll(...)`), collapse to {UNKNOWN}. Without this, recursive
    // accumulation could yield mixed sets like `{INT32, UNKNOWN}` and
    // mislead downstream consumers.
    if (ret.contains(UNKNOWN)) return EnumSet.of(UNKNOWN);
    return ret;
  }

  private Set<DType> getDTypesFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin) {
    // Dtype counterpart of the producer-cycle handling in `getShapeResultFromTensor`
    // (wala/ML#718): the engine's bottom for an unconverged dtype read is the empty set rather
    // than UNKNOWN, since an in-band UNKNOWN would collapse the caller's union to {UNKNOWN} under
    // the lattice normalization, erasing the sibling members' resolved dtypes.
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    // See memoizedShapeResult on the null-engine (outside-an-analysis) fallback.
    if (engine == null) return this.doGetDTypesFromTensor(builder, asin);
    Object key = Pair.make("dtype-delegation", asin);
    java.util.function.Supplier<Object> transfer = () -> this.doGetDTypesFromTensor(builder, asin);
    @SuppressWarnings("unchecked")
    Set<DType> diverted =
        (Set<DType>)
            (engine.isEvaluating()
                ? engine.read(key, transfer, false)
                : engine.demand(key, transfer, false));
    return diverted;
  }

  /**
   * Body of {@link #getDTypesFromTensor(PropagationCallGraphBuilder, AllocationSiteInNode)},
   * running as its engine query's transfer.
   *
   * @param builder the propagation call graph builder
   * @param asin the allocation site of the tensor
   * @return the resolved dtypes; empty when no producer resolves
   */
  private Set<DType> doGetDTypesFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    CGNode readDataNode = asin.getNode();

    // Support allocations directly in 'do' methods (preferred for 1-CFA context separation).
    if (readDataNode.getMethod().getName().toString().equals(DO_METHOD_NAME)) {
      int def = findDefinition(readDataNode, asin);
      if (def != -1) {
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(readDataNode, def);
        PointsToSetVariable defSource = null;
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }

        TensorGenerator generator =
            createManualGenerator(readDataNode, asin.concreteType().getReference(), builder);

        if (generator != null) {
          if (this.manualNode != null && this.manualNode.equals(readDataNode)) {
            return ret;
          }
          LOGGER.fine("Delegating dtype inference to: " + generator);
          Set<DType> delegated = memoizedDTypes(builder, generator);
          // A null delegation result is ⊤ (e.g., an argument dtype that resolves through neither
          // the points-to set nor the caller walk); contribute UNKNOWN rather than NPE-ing.
          ret.addAll(delegated == null ? EnumSet.of(UNKNOWN) : delegated);
        } else if (defSource != null) {
          if (this.getSource() != null && this.getSource().equals(defSource)) {
            return ret;
          }

          try {
            generator = TensorGeneratorFactory.getGenerator(defSource, builder);
          } catch (IllegalArgumentException e) {
            // Factory couldn't resolve — treat as "no generator". See wala/ML#363.
            LOGGER.log(Level.FINE, "Delegating dtype inference: factory IAE for " + defSource, e);
            generator = null;
          }
          if (generator != null) {
            LOGGER.fine("Delegating dtype inference to: " + generator);
            Set<DType> delegated = memoizedDTypes(builder, generator);
            // A null delegation result is ⊤; contribute UNKNOWN rather than NPE-ing.
            ret.addAll(delegated == null ? EnumSet.of(UNKNOWN) : delegated);
          }
        }
      }
      return ret;
    }

    // Trace back to the user-level call that invoked this generator.
    // 1. read_data is called by the operation's 'do' method; the call-graph edges identify the
    // calling instructions.
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, readDataNode)) {
      CGNode doNode = callerInvoke.fst;
      SSAAbstractInvokeInstruction call = callerInvoke.snd;

      // Construct a source for the result of this call (which is the tensor object).
      if (call.getNumberOfDefs() > 0) {
        int def = call.getDef();
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(doNode, def);
        PointsToSetVariable defSource = null;
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }

        // Try to create a manual generator for the caller (doNode) first.
        TensorGenerator generator = createManualGenerator(doNode, builder);
        if (generator == null) {
          try {
            generator = TensorGeneratorFactory.getGenerator(defSource, builder);
          } catch (IllegalArgumentException e) {
            // Factory couldn't resolve — treat as "no generator". See wala/ML#363.
            LOGGER.log(Level.FINE, "Delegating dtype inference: factory IAE for " + defSource, e);
            generator = null;
          }
        }

        if (generator != null) {
          LOGGER.fine("Delegating dtype inference to: " + generator);
          ret.addAll(memoizedDTypes(builder, generator));
        }
      }
    }
    return ret;
  }

  protected PointsToSetVariable getSource() {
    return this.source;
  }

  /**
   * Two generators are equal iff they are of the same concrete class AND share the same identity on
   * both the source and the manual-node axes. For factory-constructed generators the source is
   * populated; for manually-constructed generators the manual node is. Both fields participate in
   * the comparison so that a source-based generator and a manual-node-based generator are not
   * accidentally considered equal when they happen to share one half of the identity but not the
   * other.
   *
   * <p>Used by {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)} and its dtype
   * counterpart to avoid infinite recursion when dispatching back through {@link
   * TensorGeneratorFactory#getGenerator}. A coarser class-only equality would incorrectly conflate
   * two different generators on different value numbers or different call graph nodes — blocking
   * legitimate recursion such as nested {@link ElementWiseOperation} binop chains.
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (this.getClass() != obj.getClass()) return false;
    TensorGenerator other = (TensorGenerator) obj;
    return this.source == other.source && this.manualNode == other.manualNode;
  }

  @Override
  public int hashCode() {
    int result = this.getClass().hashCode();
    result = 31 * result + System.identityHashCode(this.source);
    result = 31 * result + System.identityHashCode(this.manualNode);
    return result;
  }

  /**
   * Resolves the points-to set of the synthetic {@code __list_append_contents__} field on the given
   * list allocation: the union of all values accumulated onto the list through {@code append}
   * (modeled in {@code PythonSSAPropagationCallGraphBuilder.processListAppend}). Element order and
   * multiplicity are not represented; generators reading elements through this helper should treat
   * the result as an unordered value union (wala/ML#570).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param listAsin The list allocation whose appended contents to read.
   * @return The union points-to set of the appended values, or {@code null} if the list has none.
   */
  protected static OrdinalSet<InstanceKey> getAppendedContentsPts(
      PropagationCallGraphBuilder builder, AllocationSiteInNode listAsin) {
    FieldReference contentsRef =
        FieldReference.findOrCreate(
            PythonTypes.Root,
            findOrCreateAsciiAtom(PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD),
            PythonTypes.Root);
    IField f = builder.getClassHierarchy().resolveField(contentsRef);
    if (f == null) return null;
    PointerKey pk = builder.getPointerKeyForInstanceField(listAsin, f);
    OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
    return pts == null || pts.isEmpty() ? null : pts;
  }

  protected CGNode getNode() {
    if (this.manualNode != null) {
      return this.manualNode;
    }
    PointerKey k = this.getSource().getPointerKey();
    if (k instanceof LocalPointerKey) {
      return ((LocalPointerKey) k).getNode();
    } else if (k instanceof ReturnValueKey) {
      return ((ReturnValueKey) k).getNode();
    }
    throw new IllegalArgumentException("Unsupported PointerKey type: " + k.getClass());
  }

  @Override
  public String toString() {
    return this.getSignature();
  }

  /**
   * Returns the TensorFlow or NumPy function signature represented by this generator.
   *
   * @return The TensorFlow or NumPy function signature represented by this generator, or {@code
   *     <unmapped:...>} if the function has no mapped signature in either {@link TensorFlowTypes}
   *     or {@link NumpyTypes}.
   */
  protected String getSignature() {
    TypeReference function;
    if (this.manualNode != null) {
      function = this.manualNode.getMethod().getDeclaringClass().getReference();
    } else {
      function = getFunction(this.getSource());
    }
    // TensorFlow signatures live in TensorFlowTypes; numpy signatures in NumpyTypes. Consult both.
    String signature =
        TYPE_REFERENCE_TO_SIGNATURE.getOrDefault(
            function, NumpyTypes.TYPE_REFERENCE_TO_SIGNATURE.get(function));
    if (signature == null) {
      return "<unmapped:" + function + ">";
    }
    return signature;
  }

  protected static final int RECEIVER_PARAMETER_POSITION = -2;

  protected static final String SELF = "self";

  protected int getArgumentValueNumber(int parameterPosition) {
    if (parameterPosition == RECEIVER_PARAMETER_POSITION)
      return this.getNode().getIR().getParameter(0);
    if (parameterPosition < 0) return UNDEFINED_PARAMETER_POSITION; // No such argument.

    int index = this.getNode().getMethod().isStatic() ? parameterPosition : parameterPosition + 1;

    if (index >= this.getNode().getIR().getNumberOfParameters())
      return UNDEFINED_PARAMETER_POSITION;

    return this.getNode().getIR().getParameter(index);
  }

  /**
   * Resolves a scalar integer argument to its static constant value, when one exists. Reads the
   * argument's points-to set (via {@link #getArgumentPointsToSet(PropagationCallGraphBuilder, int,
   * String)}, which handles the invoke-side, binary-op-side, and context-sensitive-caller cases)
   * and returns the integer carried by a {@link ConstantKey} if every key in the set agrees on the
   * same numeric value.
   *
   * <p>Returns {@code null} (signalling "not statically known") when the points-to set is empty,
   * contains a non-{@link ConstantKey} key (a runtime value — e.g. a {@code tf.shape(...)} result),
   * carries a non-numeric constant, or carries two differing numeric values. Used by generators
   * whose output shape depends on a statically-supplied integer (e.g. {@code num_segments} for the
   * unsorted-segment reductions), so the shape is recovered only when the value is genuinely a
   * compile-time constant and left at ⊤ otherwise.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The 0-based positional index of the argument (excluding {@code self}).
   * @param paramName The keyword name of the argument, or {@code null}.
   * @return The argument's static integer value, or {@code null} if it is not a single statically
   *     resolvable integer.
   */
  protected Integer resolveStaticIntArgument(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts == null || pts.isEmpty()) return null;
    Integer resolved = null;
    for (InstanceKey instanceKey : pts) {
      if (!(instanceKey instanceof ConstantKey)) return null;
      Object value = ((ConstantKey<?>) instanceKey).getValue();
      if (!(value instanceof Number)) return null;
      int candidate = ((Number) value).intValue();
      if (resolved != null && resolved != candidate) return null;
      resolved = candidate;
    }
    return resolved;
  }

  protected PythonInvokeInstruction getInvokeInstruction() {
    if (this.source == null) {
      return null;
    }
    if (this.getSource().getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.getSource().getPointerKey();
      if (lpk.getNode().equals(this.getNode())) {
        int vn = lpk.getValueNumber();
        if (vn > 0) {
          SSAInstruction def = this.getNode().getDU().getDef(vn);
          if (def instanceof PythonInvokeInstruction) {
            return (PythonInvokeInstruction) def;
          }
        }
      }
    }
    return null;
  }

  protected SSABinaryOpInstruction getBinaryOpInstruction() {
    if (this.source == null) {
      return null;
    }
    if (this.getSource().getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.getSource().getPointerKey();
      if (lpk.getNode().equals(this.getNode())) {
        int vn = lpk.getValueNumber();
        if (vn > 0) {
          SSAInstruction def = this.getNode().getDU().getDef(vn);
          if (def instanceof SSABinaryOpInstruction) {
            return (SSABinaryOpInstruction) def;
          }
        }
      }
    }
    return null;
  }

  /**
   * Returns the points-to set of the argument at the specified position or with the specified name.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @param paramName The name of the argument in the function call.
   * @return The points-to set of the argument at the specified position or with the specified name.
   */
  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    return getArgumentPointsToSet(builder, this.getNode(), paramPos, paramName);
  }

  /**
   * Resolves shapes for the argument at the given parameter position by walking the call-graph
   * edges to find each caller's invocation site, then delegating to {@link
   * #getShapes(PropagationCallGraphBuilder, CGNode, int)} with the caller's value number. This is a
   * fallback path used when {@link #getArgumentPointsToSet(PropagationCallGraphBuilder, int,
   * String)} returns an empty set because the argument's points-to set is empty — commonly the case
   * when the argument is the result of a Python binary op on tensors, for which WALA does not
   * allocate a trackable target. The caller-side recursion into {@link #getShapes(
   * PropagationCallGraphBuilder, CGNode, int)} picks up the {@link ElementWiseOperation} (or
   * similar) generator via {@link TensorGeneratorFactory}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The 0-based index of the positional parameter (excluding {@code self} for
   *     instance methods).
   * @param paramName The name of the keyword parameter, or {@code null}.
   * @return The union of shapes resolved from each caller, or {@code null} if no caller could be
   *     resolved.
   */
  protected Set<List<Dimension<?>>> getArgumentShapesViaCallers(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    Set<List<Dimension<?>>> combined = null;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn <= 0) continue;
      final int finalArgVn = argVn;
      final CGNode finalCaller = caller;
      try {
        // Exact mode (wala/ML#716): the caller walk feeds generator output computations, so the
        // per-context argument shape must be the complete set of possibilities.
        Set<List<Dimension<?>>> argShapes = this.getShapes(builder, caller, argVn, true);
        LOGGER.fine(
            () -> "getArgumentShapesViaCallers: argVn=" + finalArgVn + " shapes=" + argShapes);
        if (argShapes != null && !argShapes.isEmpty()) {
          if (combined == null) combined = HashSetFactory.make();
          combined.addAll(argShapes);
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "getArgumentShapesViaCallers: IAE for argVn="
                + finalArgVn
                + " in caller="
                + finalCaller,
            e);
      }
    }
    return combined;
  }

  /**
   * Record-carrying counterpart of {@link #getArgumentShapesViaCallers(PropagationCallGraphBuilder,
   * int, String)} (wala/ML#718): an unresolvable caller marks the union's unknown remainder and the
   * resolvable callers' members still stand.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The 0-based index of the positional parameter (excluding {@code self} for
   *     instance methods).
   * @param paramName The name of the keyword parameter, or {@code null}.
   * @return The resolution result; ⊤ when no caller resolves at all.
   */
  protected ShapeResult getArgumentShapeResultViaCallers(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    boolean callerUnknown = false;
    Set<List<Dimension<?>>> combined = null;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn <= 0) continue;
      final int finalArgVn = argVn;
      final CGNode finalCaller = caller;
      try {
        // The caller walk feeds generator output computations, so the per-context argument shape
        // must be the complete set of possibilities (wala/ML#716); a partially resolvable caller
        // keeps its members with the remainder marked (wala/ML#718).
        ShapeResult argShapes = this.getShapeResult(builder, caller, argVn, true);
        if (argShapes.hasUnknown()) callerUnknown = true;
        if (!argShapes.members().isEmpty()) {
          if (combined == null) combined = HashSetFactory.make();
          combined.addAll(argShapes.members());
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "getArgumentShapeResultViaCallers: IAE for argVn="
                + finalArgVn
                + " in caller="
                + finalCaller,
            e);
      }
    }
    return combined == null ? ShapeResult.unknown() : new ShapeResult(combined, callerUnknown);
  }

  /**
   * Returns whether the given node is a synthetic instance-method trampoline for a method of the
   * given name (e.g. the {@code build} or {@code call} trampoline of a Keras layer).
   *
   * @param node The {@link CGNode} to test.
   * @param methodName The trampolined method's Python-level name.
   * @return {@code true} iff {@code node} is a trampoline whose target method has the given name.
   */
  protected static boolean isTrampolineFor(CGNode node, String methodName) {
    return node.getMethod().getName().toString().startsWith("trampoline")
        && node.getMethod()
            .getDeclaringClass()
            .getReference()
            .getName()
            .toString()
            .endsWith("/" + methodName);
  }

  /**
   * Returns the calling invocations of the given node: for each call-graph predecessor, the invoke
   * instructions at the sites that can dispatch to it. Derived from call-graph edges rather than a
   * {@code CALL_STRING} lookup, so it works for any {@link com.ibm.wala.ipa.callgraph.Context}
   * shape — the receiver-keyed contexts of <a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a> carry no call string.
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph to consult.
   * @param node The callee {@link CGNode}.
   * @return The (caller, invoke) pairs whose invocations can dispatch to the given node.
   */
  protected static List<Pair<CGNode, SSAAbstractInvokeInstruction>> getCallerInvokes(
      PropagationCallGraphBuilder builder, CGNode node) {
    List<Pair<CGNode, SSAAbstractInvokeInstruction>> ret = new ArrayList<>();
    CallGraph callGraph = builder.getCallGraph();
    for (Iterator<CGNode> it = callGraph.getPredNodes(node); it.hasNext(); ) {
      CGNode caller = it.next();
      if (caller.getIR() == null) continue;
      for (Iterator<CallSiteReference> sites = callGraph.getPossibleSites(caller, node);
          sites.hasNext(); )
        for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(sites.next()))
          ret.add(Pair.make(caller, call));
    }
    return ret;
  }

  /**
   * Recovers an allocator's shape when its shape argument is another tensor's {@code .shape} (e.g.
   * {@code tf.ones(x.shape)}). Such an argument is a {@link PythonPropertyRead} of member {@code
   * "shape"} whose points-to set is empty, so ordinary shape resolution falls through to {@link
   * #getDefaultShapes}; this resolves the shape of the underlying tensor instead of dropping to ⊤.
   * Walks the call-graph callers (the allocation's synthetic node has no argument IR of its own) to
   * find the call site and the shape argument's value number, mirroring {@link
   * #getArgumentShapesViaCallers}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param paramPos The 0-based positional index of the shape parameter (excluding {@code self}).
   * @param paramName The shape parameter's keyword name, or {@code null}.
   * @return The union of the source tensors' shapes, or {@code null} if no {@code .shape} argument
   *     is found or none resolves.
   */
  protected Set<List<Dimension<?>>> getShapeFromShapeAttributeArgument(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    Set<List<Dimension<?>>> combined = null;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn <= 0) continue;
      SSAInstruction def = caller.getDU().getDef(argVn);
      if (!(def instanceof PythonPropertyRead)) continue;
      PythonPropertyRead propRead = (PythonPropertyRead) def;
      SymbolTable st = caller.getIR().getSymbolTable();
      int memberVn = propRead.getMemberRef();
      if (!st.isStringConstant(memberVn) || !"shape".equals(st.getStringValue(memberVn))) continue;
      int tensorVn = propRead.getObjectRef();
      try {
        Set<List<Dimension<?>>> tensorShapes = this.getShapes(builder, caller, tensorVn);
        if (tensorShapes != null && !tensorShapes.isEmpty()) {
          if (combined == null) combined = HashSetFactory.make();
          combined.addAll(tensorShapes);
        }
      } catch (IllegalArgumentException e) {
        // The source tensor's shape couldn't be resolved; fall through to ⊤.
        LOGGER.log(
            Level.FINE,
            "getShapeFromShapeAttributeArgument: could not resolve .shape source tensorVn="
                + tensorVn
                + " in caller="
                + caller,
            e);
      }
    }
    return combined;
  }

  /**
   * Resolves a shape argument that is a <em>shape vector</em>: a Python value derived from a
   * tensor's shape through the {@code t.shape} / {@code t.shape.as_list()} / {@code
   * t.shape.as_list()[a:b]} idioms (<a
   * href="https://github.com/wala/ML/issues/703">wala/ML#703</a>). Such values have an empty
   * points-to set (the {@code shape} member and {@code as_list} are unmodeled), so ordinary shape
   * resolution can't see them; this recovers the source tensor's shape by def-use walking and
   * applies any slice with constant bounds.
   *
   * <p>Tries the generator's own invoke first (caller-side sources); for synthetic/manual nodes it
   * walks the call-graph callers, mirroring {@link #getShapeFromShapeAttributeArgument}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param paramPos The 0-based positional index of the shape parameter (excluding {@code self}).
   * @param paramName The shape parameter's keyword name, or {@code null}.
   * @return The union of the recovered (sub-)shapes, or {@code null} if the argument isn't a
   *     resolvable shape vector.
   */
  protected Set<List<Dimension<?>>> getShapesFromShapeVectorArgument(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    return this.getShapeResultFromShapeVectorArgument(builder, paramPos, paramName).toLegacy();
  }

  /**
   * Record-carrying core of {@link #getShapesFromShapeVectorArgument(PropagationCallGraphBuilder,
   * int, String)} (wala/ML#718): a partially resolvable shape vector keeps its members with the
   * remainder marked.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param paramPos The 0-based positional index of the shape parameter (excluding {@code self}).
   * @param paramName The shape parameter's keyword name, or {@code null}.
   * @return The resolution result.
   */
  protected ShapeResult getShapeResultFromShapeVectorArgument(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int argVn = -1;
      if (paramName != null) argVn = call.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
        int numPosParams = call.getNumberOfUses() - 1 - numKeywords;
        if (paramPos < numPosParams) argVn = call.getUse(paramPos + 1);
      }
      if (argVn > 0) return this.getShapeResultOfShapeVector(builder, this.getNode(), argVn);
      return ShapeResult.unknown();
    }

    // Synthetic/manual node: walk the call-graph callers to find the argument's value number.
    boolean callerUnknown = false;
    Set<List<Dimension<?>>> combined = null;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn <= 0) continue;
      ShapeResult shapes = this.getShapeResultOfShapeVector(builder, caller, argVn);
      if (shapes.hasUnknown()) callerUnknown = true;
      if (!shapes.members().isEmpty()) {
        if (combined == null) combined = HashSetFactory.make();
        combined.addAll(shapes.members());
      }
    }
    return combined == null ? ShapeResult.unknown() : new ShapeResult(combined, callerUnknown);
  }

  /**
   * Def-use walker behind {@link #getShapesFromShapeVectorArgument}: resolves a value number to the
   * tensor (sub-)shape it denotes. Recognizes, recursively:
   *
   * <ul>
   *   <li>{@code t.shape} — a {@link PythonPropertyRead} of member {@code "shape"}; resolves the
   *       source tensor's shapes via {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)}.
   *   <li>{@code v.as_list()} — an invoke whose function object is a property read of member {@code
   *       "as_list"}; recurses on the receiver.
   *   <li>{@code v[a:b]} — an invoke of the {@code slice} builtin with constant (or {@code None})
   *       bounds and a unit step; recurses on the receiver and takes the corresponding sub-list of
   *       each shape, with Python negative-index semantics.
   * </ul>
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to resolve.
   * @return The union of the denoted (sub-)shapes, or {@code null} (⊤) when the def-use chain
   *     doesn't match a recognized shape-vector form, a slice bound isn't a compile-time constant
   *     (the sub-list's rank is then unknown), or the source tensor's shape doesn't resolve.
   */
  protected Set<List<Dimension<?>>> getShapesOfShapeVector(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    return this.getShapeResultOfShapeVector(builder, node, vn).toLegacy();
  }

  /**
   * Record-carrying variant of {@link #getShapesOfShapeVector(PropagationCallGraphBuilder, CGNode,
   * int)} (wala/ML#718): a partially resolvable source tensor keeps its resolvable members walking
   * through the recognized forms while the unknown remainder rides along.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to resolve.
   * @return The resolution result.
   */
  protected ShapeResult getShapeResultOfShapeVector(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    return getShapeResultOfShapeVector(builder, node, vn, HashSetFactory.make());
  }

  /**
   * Core of {@link #getShapeResultOfShapeVector(PropagationCallGraphBuilder, CGNode, int)}
   * threading the set of callee nodes already on the walk, guarding recursive helpers
   * (wala/ML#706).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to resolve.
   * @param visited The callee nodes already on the walk.
   * @return The resolution result; see the wrapper.
   */
  private ShapeResult getShapeResultOfShapeVector(
      PropagationCallGraphBuilder builder, CGNode node, int vn, Set<CGNode> visited) {
    if (vn <= 0 || node.getDU() == null || node.getIR() == null) return ShapeResult.unknown();
    SSAInstruction def = node.getDU().getDef(vn);
    SymbolTable st = node.getIR().getSymbolTable();

    // The `input_shape` parameter of a Keras `build` method denotes the shape of the layer-call
    // input, but the lazy-build machinery invokes `build` through the method's own trampoline
    // with `self` as the only argument (wala/ML#595), so the parameter never receives a shape in
    // the PA. Resolve it via the callers: the build body's caller is the build trampoline, whose
    // own caller is either the layer-call trampoline (the lazy-build pattern — its second
    // parameter is the layer call's first positional argument, the input tensor) or a real method
    // body issuing an explicit `x.build(shape)` call, whose argument is walked as a shape vector
    // (wala/ML#712).
    if (def == null
        && node.getMethod()
            .getDeclaringClass()
            .getReference()
            .getName()
            .toString()
            .endsWith("/" + KERAS_BUILD_METHOD_NAME)
        && node.getMethod().getNumberOfParameters() >= 3
        && vn == node.getIR().getParameter(2)) {
      LOGGER.fine(
          () ->
              "getShapesOfShapeVector: resolving the build input_shape parameter in "
                  + describe(node));
      boolean callerUnknown = false;
      Set<List<Dimension<?>>> combined = null;
      for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
          getCallerInvokes(builder, node)) {
        CGNode buildTrampoline = callerInvoke.fst;
        if (!isTrampolineFor(buildTrampoline, KERAS_BUILD_METHOD_NAME))
          return ShapeResult.unknown();
        for (Pair<CGNode, SSAAbstractInvokeInstruction> outerInvoke :
            getCallerInvokes(builder, buildTrampoline)) {
          CGNode outerCaller = outerInvoke.fst;
          SSAAbstractInvokeInstruction outerCall = outerInvoke.snd;
          ShapeResult shapes = ShapeResult.unknown();
          if (isTrampolineFor(outerCaller, CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)
              && outerCaller.getIR() != null
              && outerCaller.getMethod().getNumberOfParameters() >= 2) {
            try {
              shapes =
                  this.getShapeResult(
                      builder, outerCaller, outerCaller.getIR().getParameter(1), true);
            } catch (IllegalArgumentException e) {
              LOGGER.log(
                  Level.FINE,
                  "getShapesOfShapeVector: could not resolve the layer-call input via caller="
                      + describe(outerCaller),
                  e);
            }
          } else if (outerCall.getNumberOfUses() >= 2 && visited.add(outerCaller))
            // An explicit `x.build(shape)` call: walk the actual argument.
            shapes =
                this.getShapeResultOfShapeVector(
                    builder, outerCaller, outerCall.getUse(1), visited);
          // The parameter unions all callers; an unresolved caller marks the unknown remainder
          // and the resolvable callers' members still stand (wala/ML#718).
          if (shapes.hasUnknown() || shapes.isBottom()) callerUnknown = true;
          if (!shapes.members().isEmpty()) {
            if (combined == null) combined = HashSetFactory.make();
            combined.addAll(shapes.members());
          }
        }
      }
      return combined == null ? ShapeResult.unknown() : new ShapeResult(combined, callerUnknown);
    }

    if (def instanceof PythonPropertyRead) {
      PythonPropertyRead read = (PythonPropertyRead) def;
      int memberVn = read.getMemberRef();
      if (st.isStringConstant(memberVn) && "shape".equals(st.getStringValue(memberVn))) {
        try {
          // The walk's folds assert per-member facts (subscripts, slices, products), so the
          // source must be the complete set of possibilities; a partially resolvable source
          // keeps its members with the remainder marked (wala/ML#716, wala/ML#718).
          return this.getShapeResult(builder, node, read.getObjectRef(), true);
        } catch (IllegalArgumentException e) {
          LOGGER.log(
              Level.FINE,
              "getShapesOfShapeVector: could not resolve .shape source in node=" + node,
              e);
          return ShapeResult.unknown();
        }
      }
      return ShapeResult.unknown();
    }

    if (def instanceof PythonInvokeInstruction) {
      PythonInvokeInstruction invoke = (PythonInvokeInstruction) def;
      if (invoke.getNumberOfUses() < 1) return ShapeResult.unknown();
      SSAInstruction funcDef = node.getDU().getDef(invoke.getUse(0));

      // tf.shape(x): a shape vector by construction — element i is the runtime size of x's axis
      // i — so the vector's members are exactly x's shapes, classified per axis on both sides of
      // the wala/ML#721 criterion: a `None` axis contributes its DynamicDim (the dynamic
      // evidence wala/ML#722's over-capture ignored), a statically known axis its constant
      // (TensorFlow itself constant-folds tf.shape over static axes), an unresolved axis its
      // UnresolvedDim.
      if (dispatchesToTfShape(builder, node, invoke) && invoke.getNumberOfUses() >= 2) {
        try {
          return this.getShapeResult(builder, node, invoke.getUse(1), true);
        } catch (IllegalArgumentException e) {
          LOGGER.log(
              Level.FINE,
              "getShapesOfShapeVector: could not resolve the tf.shape input in node=" + node,
              e);
          return ShapeResult.unknown();
        }
      }

      // v.as_list(): peel the call and recurse on the receiver of the property read.
      if (funcDef instanceof PythonPropertyRead) {
        PythonPropertyRead funcRead = (PythonPropertyRead) funcDef;
        int memberVn = funcRead.getMemberRef();
        if (st.isStringConstant(memberVn) && "as_list".equals(st.getStringValue(memberVn)))
          return this.getShapeResultOfShapeVector(builder, node, funcRead.getObjectRef(), visited);
        return ShapeResult.unknown();
      }

      // v[a:b]: a slice-builtin invoke of the form slice(receiver, lower, upper, step).
      if (dispatchesToSliceBuiltin(builder, node, invoke) && invoke.getNumberOfUses() >= 2) {
        ShapeResult base =
            this.getShapeResultOfShapeVector(builder, node, invoke.getUse(1), visited);
        if (base.members().isEmpty()) return ShapeResult.unknown();

        // A nested slice object as a bound means a multi-dim subscript; a shape vector is 1-D, so
        // that form isn't a shape-list slice.
        for (int u = 2; u < invoke.getNumberOfUses(); u++)
          if (isSliceObjectDef(node, invoke.getUse(u))) return ShapeResult.unknown();

        Integer lower = sliceBoundOrNull(builder, node, st, invoke, 2);
        Integer upper = sliceBoundOrNull(builder, node, st, invoke, 3);
        Integer step = sliceBoundOrNull(builder, node, st, invoke, 4);
        if (Objects.equals(lower, UNRESOLVED_BOUND)
            || Objects.equals(upper, UNRESOLVED_BOUND)
            || Objects.equals(step, UNRESOLVED_BOUND))
          return ShapeResult.unknown(); // Non-constant bound: the sub-list's rank is unknown.
        // A constant positive step strides the bounded range; a non-positive one keeps the ⊤
        // fallback, covering negative steps (which also reverse) and the invalid zero
        // (wala/ML#709).
        int stride = step == null ? 1 : step;
        if (stride < 1) return ShapeResult.unknown();

        // Slice per member; the base's unknown remainder rides through (wala/ML#718).
        Set<List<Dimension<?>>> sliced = HashSetFactory.make();
        for (List<Dimension<?>> dims : base.members()) {
          int n = dims.size();
          int from = lower == null ? 0 : lower < 0 ? Math.max(0, n + lower) : Math.min(lower, n);
          int to = upper == null ? n : upper < 0 ? Math.max(0, n + upper) : Math.min(upper, n);
          List<Dimension<?>> taken = new ArrayList<>();
          for (int i = from; i < to; i += stride) taken.add(dims.get(i));
          sliced.add(taken);
        }
        return new ShapeResult(sliced, base.hasUnknown());
      }

      // A call to a user helper (the BERT/ALBERT get_shape_list pattern): follow each callee's
      // returned value. The callee's parameters carry the caller's arguments in their points-to
      // sets (the PA is interprocedural), so the .shape base case resolves across the boundary
      // without an explicit parameter-to-argument mapping (wala/ML#706).
      return this.shapeResultOfCalleeReturns(builder, node, invoke, visited);
    }

    // a + b: the concatenation of two shape vectors, e.g. `batch_dims + outer_dims`
    // (wala/ML#708). Either operand may also be a literal list of resolvable elements
    // (`batch_dims + [inner_dim]`). Either operand's unknown remainder rides through the
    // member cross-product (wala/ML#718).
    if (def instanceof SSABinaryOpInstruction
        && ((SSABinaryOpInstruction) def).getOperator() == IBinaryOpInstruction.Operator.ADD) {
      ShapeResult left =
          this.shapeResultOfShapeVectorOrLiteralList(builder, node, def.getUse(0), visited);
      if (left.members().isEmpty()) return ShapeResult.unknown();
      ShapeResult right =
          this.shapeResultOfShapeVectorOrLiteralList(builder, node, def.getUse(1), visited);
      if (right.members().isEmpty()) return ShapeResult.unknown();
      Set<List<Dimension<?>>> concatenated = HashSetFactory.make();
      for (List<Dimension<?>> l : left.members())
        for (List<Dimension<?>> r : right.members()) {
          List<Dimension<?>> joined = new ArrayList<>(l);
          joined.addAll(r);
          concatenated.add(joined);
        }
      return new ShapeResult(concatenated, left.hasUnknown() || right.hasUnknown());
    }
    return ShapeResult.unknown();
  }

  /**
   * Resolves a concatenation operand: a shape vector per {@link #getShapesOfShapeVector}, or a
   * literal list construction whose integer-indexed elements each resolve to a constant (including
   * the {@code np.prod} fold) or {@code None} (wala/ML#708).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The operand's value number.
   * @param visited The callee nodes already on the walk.
   * @return The operand's (sub-)shapes, or {@code null} (⊤) when it resolves as neither form.
   */
  private Set<List<Dimension<?>>> shapesOfShapeVectorOrLiteralList(
      PropagationCallGraphBuilder builder, CGNode node, int vn, Set<CGNode> visited) {
    return this.shapeResultOfShapeVectorOrLiteralList(builder, node, vn, visited).toLegacy();
  }

  /**
   * Record-carrying core of {@link #shapesOfShapeVectorOrLiteralList(PropagationCallGraphBuilder,
   * CGNode, int, Set)} (wala/ML#718): a partially resolvable shape-vector operand keeps its members
   * with the remainder marked; the literal-list form resolves fully or not at all.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The operand's value number.
   * @param visited The callee nodes already on the walk.
   * @return The resolution result.
   */
  private ShapeResult shapeResultOfShapeVectorOrLiteralList(
      PropagationCallGraphBuilder builder, CGNode node, int vn, Set<CGNode> visited) {
    ShapeResult shapes = this.getShapeResultOfShapeVector(builder, node, vn, visited);
    if (!shapes.members().isEmpty()) return shapes;

    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof SSANewInstruction)) return ShapeResult.unknown();
    TypeReference allocated = ((SSANewInstruction) def).getNewSite().getDeclaredType();
    if (!allocated.equals(list) && !allocated.equals(tuple)) return ShapeResult.unknown();

    SymbolTable st = node.getIR().getSymbolTable();
    TreeMap<Integer, Set<Dimension<?>>> byIndex = new TreeMap<>();
    for (Iterator<SSAInstruction> uses = node.getDU().getUses(vn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      int objRef;
      int writtenVal;
      int memberVn;
      if (use instanceof PythonPropertyWrite) {
        PythonPropertyWrite w = (PythonPropertyWrite) use;
        objRef = w.getObjectRef();
        writtenVal = w.getValue();
        memberVn = w.getMemberRef();
      } else if (use instanceof SSAPutInstruction && !((SSAPutInstruction) use).isStatic()) {
        SSAPutInstruction w = (SSAPutInstruction) use;
        if (w.getRef() != vn) continue; // A write to some other object.
        writtenVal = w.getVal();
        try {
          byIndex.put(
              Integer.parseInt(w.getDeclaredField().getName().toString()),
              elementDims(builder, node, st, writtenVal));
        } catch (NumberFormatException e) {
          // Not an index write.
        }
        continue;
      } else continue;
      if (objRef != vn) continue;
      Integer index = null;
      if (st.isNumberConstant(memberVn))
        index = ((Number) st.getConstantValue(memberVn)).intValue();
      else if (st.isStringConstant(memberVn)) {
        try {
          index = Integer.parseInt(st.getStringValue(memberVn));
        } catch (NumberFormatException e) {
          // Not an index write.
        }
      }
      if (index == null) continue;
      byIndex.put(index, elementDims(builder, node, st, writtenVal));
    }
    if (byIndex.isEmpty() || byIndex.containsValue(null)) return ShapeResult.unknown();
    // The indices must be contiguous from 0 or the element order can't be reconstructed.
    if (byIndex.firstKey() != 0 || byIndex.lastKey() != byIndex.size() - 1)
      return ShapeResult.unknown();

    // Cross-product the per-position options (a multi-member source shape contributes several,
    // wala/ML#717); a blowup degrades to ⊤ rather than an unbounded union.
    List<List<Dimension<?>>> composed = new ArrayList<>();
    composed.add(new ArrayList<>());
    for (Set<Dimension<?>> options : byIndex.values()) {
      List<List<Dimension<?>>> next = new ArrayList<>(composed.size() * options.size());
      for (List<Dimension<?>> prefix : composed)
        for (Dimension<?> option : options) {
          List<Dimension<?>> extended = new ArrayList<>(prefix);
          extended.add(option);
          next.add(extended);
        }
      if (next.size() > 16) return ShapeResult.unknown();
      composed = next;
    }
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.addAll(composed);
    return ShapeResult.of(ret);
  }

  /**
   * Resolves one literal-list element to a dimension: a constant integer becomes a {@link
   * NumericDim}, {@code None} becomes {@link DynamicDim}, and an {@code np.prod} over a resolvable
   * shape vector folds (wala/ML#708).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The element's value number.
   * @return The element's dimension, or {@code null} when it doesn't resolve.
   */
  private Dimension<?> elementDim(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    Dimension<?> prod = this.prodOfShapeVectorDim(builder, node, st, vn);
    if (prod != null) return prod;
    Dimension<?> subscript = this.resolveShapeVectorElementDim(builder, node, st, vn);
    if (subscript != null) return subscript;
    Dimension<?> arithmetic = this.resolveArithmeticOverSubscriptDim(builder, node, st, vn);
    if (arithmetic != null) return arithmetic;
    Dimension<?> attribute = this.resolveStoredAttributeDim(builder, node, vn);
    if (attribute != null) return attribute;
    Integer constant = constantIntValueOrSentinel(builder, node, st, vn);
    if (constant == null) return DynamicDim.INSTANCE; // None: the dynamic-dim marker.
    if (Objects.equals(constant, UNRESOLVED_BOUND)) return null;
    return new NumericDim(constant);
  }

  /**
   * Multi-member counterpart of {@link #elementDim}: when the element's source shape vector is a φ
   * of several shapes (e.g. the rank-conditional {@code tf.expand_dims} guard of NLPGNN's {@code
   * WDEmbedding.call}), returns one dimension option per member instead of bailing on the
   * ambiguity; the literal-list composition cross-products the options (wala/ML#717). Mixed
   * pairings the cross-product introduces are a sound over-approximation.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The element's value number.
   * @return The element's dimension options, or {@code null} when it doesn't resolve.
   */
  private Set<Dimension<?>> elementDims(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    Set<Dimension<?>> subscripts = this.resolveShapeVectorElementDims(builder, node, st, vn);
    if (subscripts != null) return subscripts;
    Set<Dimension<?>> arithmetic = this.resolveArithmeticOverSubscriptDims(builder, node, st, vn);
    if (arithmetic != null) return arithmetic;
    Set<Dimension<?>> products = this.prodOfShapeVectorDims(builder, node, st, vn);
    if (products != null) return products;
    Dimension<?> single = this.elementDim(builder, node, st, vn);
    return single == null ? null : Collections.singleton(single);
  }

  /**
   * Multi-member counterpart of {@link #resolveShapeVectorElementDim}: resolves the subscript per
   * source-shape member, skipping members the index cannot address (those cannot be the runtime
   * shape, since the subscript would fail on them).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number of the candidate subscript result.
   * @return The denoted dimension options, or {@code null} when the value isn't a constant-indexed
   *     subscript of a resolvable shape vector with fewer than two members addressable.
   */
  private Set<Dimension<?>> resolveShapeVectorElementDims(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0) return null;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof PythonPropertyRead)) return null;
    PythonPropertyRead read = (PythonPropertyRead) def;
    Integer index = subscriptIndexOrNull(builder, node, st, read.getMemberRef());
    if (index == null) return null;

    Set<List<Dimension<?>>> shapes =
        this.getShapesOfShapeVector(builder, node, read.getObjectRef());
    if (shapes == null || shapes.size() < 2) return null; // The singleton path covers size 1.
    Set<Dimension<?>> options = HashSetFactory.make();
    for (List<Dimension<?>> dims : shapes) {
      int k = index < 0 ? dims.size() + index : index;
      if (k < 0 || k >= dims.size()) continue;
      options.add(dims.get(k));
    }
    return options.isEmpty() ? null : options;
  }

  /**
   * Multi-member counterpart of {@link #resolveArithmeticOverSubscriptDim}: folds the binary op
   * over the cross-product of the operands' options, so a subscript over a multi-member source
   * yields one result option per member.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number of the candidate expression result.
   * @return The folded dimension options, or {@code null} when the expression doesn't resolve as a
   *     multi-option fold (the singleton path covers the unambiguous case).
   */
  private Set<Dimension<?>> resolveArithmeticOverSubscriptDims(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0 || node.getDU() == null) return null;
    int peeled = peelIntBuiltin(builder, node, vn);
    boolean truncated = peeled != vn;
    SSAInstruction def = node.getDU().getDef(peeled);
    if (!(def instanceof SSABinaryOpInstruction)) return null;
    SSABinaryOpInstruction binOp = (SSABinaryOpInstruction) def;

    Set<Integer> lefts = this.scalarOperandOptionsOrNull(builder, node, st, binOp.getUse(0));
    Set<Integer> rights = this.scalarOperandOptionsOrNull(builder, node, st, binOp.getUse(1));
    if (lefts == null || rights == null) {
      // Mirror of the singleton path's degradation (wala/ML#717): dimension arithmetic with an
      // unresolvable co-operand is one scalar dimension of unknown fixed value (wala/ML#721).
      if ((lefts == null) != (rights == null)
          && this.derivesFromShapeVector(
              builder, node, st, lefts == null ? binOp.getUse(1) : binOp.getUse(0)))
        return Collections.singleton(UnresolvedDim.INSTANCE);
      return null;
    }
    if (lefts.size() < 2 && rights.size() < 2) return null; // The singleton path covers this.

    Set<Dimension<?>> options = HashSetFactory.make();
    for (int left : lefts)
      for (int right : rights) {
        if (binOp.getOperator() == IBinaryOpInstruction.Operator.DIV
            && !truncated
            && (right == 0 || left % right != 0)) {
          options.add(UnresolvedDim.INSTANCE); // Non-exact bare float division: fixed value.
          continue;
        }
        Integer value = TensorType.applyIntBinOp(binOp.getOperator(), left, right);
        options.add(value == null ? UnresolvedDim.INSTANCE : new NumericDim(value));
      }
    return options.isEmpty() ? null : options;
  }

  /**
   * Multi-member counterpart of {@link #scalarOperandOrNull}: the operand's possible integer
   * values, one per source-shape member for a subscript operand.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The operand's value number.
   * @return The operand's integer options, or {@code null} when it doesn't resolve or a member is
   *     non-numeric (the set could not cover every runtime possibility).
   */
  private Set<Integer> scalarOperandOptionsOrNull(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    Integer single = this.scalarOperandOrNull(builder, node, st, vn);
    if (single != null) return Collections.singleton(single);

    int peeled = peelIntBuiltin(builder, node, vn);
    Set<Dimension<?>> dims = this.resolveShapeVectorElementDims(builder, node, st, peeled);
    if (dims == null) return null;
    Set<Integer> options = HashSetFactory.make();
    for (Dimension<?> dim : dims) {
      if (!(dim instanceof NumericDim)) return null; // A non-numeric member: not enumerable.
      options.add(((NumericDim) dim).value());
    }
    return options.isEmpty() ? null : options;
  }

  /**
   * Resolves a subscript's member reference to a constant index, accepting a numeric constant, an
   * integer-parsable string constant, or a resolvable expression.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for PA lookup.
   * @param node The {@link CGNode} whose IR defines the member reference.
   * @param st The node's symbol table.
   * @param memberVn The member reference's value number.
   * @return The index, or {@code null} when it doesn't resolve.
   */
  private static Integer subscriptIndexOrNull(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int memberVn) {
    if (st.isStringConstant(memberVn)) {
      try {
        return Integer.parseInt(st.getStringValue(memberVn));
      } catch (NumberFormatException e) {
        return null; // An attribute read, not a subscript.
      }
    }
    Integer index = constantIntValueOrSentinel(builder, node, st, memberVn);
    return index == null || Objects.equals(index, UNRESOLVED_BOUND) ? null : index;
  }

  /**
   * Returns whether the invoke dispatches to the {@code slice} builtin, either syntactically (its
   * function object is a {@link PythonTypes#SLICE_BUILTIN} allocation) or via the call graph. The
   * latter covers slices inside nested functions, where the builtin's function object is a lexical
   * read rather than a local allocation (wala/ML#708).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the targets.
   * @param node The calling {@link CGNode}.
   * @param invoke The invoke instruction to test.
   * @return {@code true} iff the invoke dispatches to the slice builtin.
   */
  private static boolean dispatchesToSliceBuiltin(
      PropagationCallGraphBuilder builder, CGNode node, PythonInvokeInstruction invoke) {
    SSAInstruction funcDef = node.getDU().getDef(invoke.getUse(0));
    if (funcDef instanceof SSANewInstruction
        && ((SSANewInstruction) funcDef)
            .getNewSite()
            .getDeclaredType()
            .equals(PythonTypes.SLICE_BUILTIN)) return true;
    Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
    if (targets == null || targets.isEmpty()) return false;
    for (CGNode target : targets)
      if (!target.getMethod().getDeclaringClass().getReference().equals(PythonTypes.SLICE_BUILTIN))
        return false;
    return true;
  }

  /**
   * Returns whether the invoke dispatches to {@code tf.shape} via the call graph (wala/ML#722).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the targets.
   * @param node The calling {@link CGNode}.
   * @param invoke The invoke instruction to test.
   * @return {@code true} iff every resolved target is {@code tf.shape}'s synthetic method.
   */
  private static boolean dispatchesToTfShape(
      PropagationCallGraphBuilder builder, CGNode node, PythonInvokeInstruction invoke) {
    Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
    if (targets == null || targets.isEmpty()) return false;
    for (CGNode target : targets)
      if (!target
          .getMethod()
          .getDeclaringClass()
          .getReference()
          .equals(TensorFlowTypes.SHAPE_OF.getDeclaringClass())) return false;
    return true;
  }

  /**
   * Follows an invoke into each callee the call graph resolves for it and walks every returned
   * value's def-use chain via {@link #getShapesOfShapeVector}, unioning the results (wala/ML#706).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the targets.
   * @param node The calling {@link CGNode}.
   * @param invoke The invoke instruction to follow.
   * @param visited The callee nodes already on the walk, guarding recursive helpers.
   * @return The resolution result: ⊤ when there are no resolvable targets or a return resolves to
   *     nothing at all; a partially resolvable return keeps its members with the remainder marked
   *     (wala/ML#718).
   */
  private ShapeResult shapeResultOfCalleeReturns(
      PropagationCallGraphBuilder builder,
      CGNode node,
      PythonInvokeInstruction invoke,
      Set<CGNode> visited) {
    Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
    if (targets == null || targets.isEmpty()) return ShapeResult.unknown();

    boolean hasUnknown = false;
    Set<List<Dimension<?>>> combined = HashSetFactory.make();
    for (CGNode callee : targets) {
      if (callee.getIR() == null || callee.getDU() == null) return ShapeResult.unknown();
      if (!visited.add(callee)) return ShapeResult.unknown(); // Recursive helper: unmodeled.
      try {
        boolean sawReturn = false;
        for (Iterator<SSAInstruction> it = callee.getIR().iterateAllInstructions();
            it.hasNext(); ) {
          SSAInstruction instruction = it.next();
          if (!(instruction instanceof SSAReturnInstruction)) continue;
          SSAReturnInstruction ret = (SSAReturnInstruction) instruction;
          // A bare `return` means the helper can produce no value (None) on some path, so the
          // call's result isn't a resolvable shape vector.
          if (ret.getResult() < 0) return ShapeResult.unknown();
          sawReturn = true;
          ShapeResult shapes =
              this.getShapeResultOfShapeVector(builder, callee, ret.getResult(), visited);
          // Every return must resolve; an unresolvable remainder is marked rather than
          // collapsing the resolvable members (wala/ML#718).
          if (shapes.members().isEmpty()) return ShapeResult.unknown();
          if (shapes.hasUnknown()) hasUnknown = true;
          combined.addAll(shapes.members());
        }
        if (!sawReturn) return ShapeResult.unknown();
      } finally {
        visited.remove(callee);
      }
    }
    return combined.isEmpty() ? ShapeResult.unknown() : new ShapeResult(combined, hasUnknown);
  }

  /**
   * Tensor-value twin of {@link #shapeResultOfCalleeReturns}: follows an invoke into each callee
   * the call graph resolves for it and reads every returned value's shape via {@link
   * #getShapeResult(PropagationCallGraphBuilder, CGNode, int, boolean)}, unioning the results
   * (wala/ML#718). Unlike the shape-vector walk, whose folds assert per-member facts and must
   * resolve every return, a tensor-value read is partial-friendly: an unresolvable return (or a
   * bare {@code return}, whose {@code None} is not a tensor) marks the unknown remainder while the
   * resolvable returns' members stand.
   *
   * <p>Recursion through cyclic call chains is bounded by the worklist engine, which answers a
   * re-entered {@code (node, vn)} read with its current state and converges the cycle.
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the targets.
   * @param node The calling {@link CGNode}.
   * @param invoke The invoke instruction whose result is being read.
   * @param exact Whether the read is exact (see {@link #getShapeResult(PropagationCallGraphBuilder,
   *     CGNode, int, boolean)}).
   * @return The union of the callees' return-value shapes; ⊤ when nothing resolves.
   */
  private ShapeResult shapeResultOfCalleeReturnValues(
      PropagationCallGraphBuilder builder,
      CGNode node,
      SSAAbstractInvokeInstruction invoke,
      boolean exact) {
    Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
    if (targets == null || targets.isEmpty()) return ShapeResult.unknown();

    boolean hasUnknown = false;
    Set<List<Dimension<?>>> combined = HashSetFactory.make();
    for (CGNode callee : targets) {
      if (callee.getIR() == null || callee.getDU() == null) return ShapeResult.unknown();
      boolean sawReturn = false;
      for (Iterator<SSAInstruction> it = callee.getIR().iterateAllInstructions(); it.hasNext(); ) {
        SSAInstruction instruction = it.next();
        if (!(instruction instanceof SSAReturnInstruction)) continue;
        SSAReturnInstruction ret = (SSAReturnInstruction) instruction;
        sawReturn = true;
        if (ret.getResult() < 0) {
          hasUnknown = true;
          continue;
        }
        ShapeResult shapes;
        try {
          shapes = this.getShapeResult(builder, callee, ret.getResult(), exact);
        } catch (IllegalArgumentException e) {
          LOGGER.log(
              Level.FINE,
              e,
              () -> "shapeResultOfCalleeReturnValues: IAE reading a return of " + describe(callee));
          hasUnknown = true;
          continue;
        }
        if (shapes.hasUnknown() || shapes.isBottom()) hasUnknown = true;
        combined.addAll(shapes.members());
      }
      if (!sawReturn) hasUnknown = true;
    }
    return combined.isEmpty() ? ShapeResult.unknown() : new ShapeResult(combined, hasUnknown);
  }

  /**
   * Returns whether this generator's shape argument is structurally a shape vector (see {@link
   * #isShapeVectorChain}), regardless of whether the provenance walk can resolve it. Consumers use
   * this to distinguish "the shape is determined by a shape vector we couldn't resolve" (output is
   * ⊤) from "no shape information at all" (an input-derived fallback may apply). Mirrors {@link
   * #getShapesFromShapeVectorArgument}'s direct-invoke and caller-walk arms.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph lookup.
   * @param paramPos The 0-based positional index of the shape parameter (excluding {@code self}).
   * @param paramName The shape parameter's keyword name, or {@code null}.
   * @return {@code true} iff the shape argument's def-use chain matches a shape-vector form.
   */
  protected boolean isShapeVectorArgument(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int argVn = -1;
      if (paramName != null) argVn = call.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
        int numPosParams = call.getNumberOfUses() - 1 - numKeywords;
        if (paramPos < numPosParams) argVn = call.getUse(paramPos + 1);
      }
      return argVn > 0 && isShapeVectorChain(builder, this.getNode(), argVn);
    }
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn > 0 && isShapeVectorChain(builder, caller, argVn)) return true;
    }
    return false;
  }

  /**
   * Structural companion of {@link #getShapesOfShapeVector}: returns whether the value's def-use
   * chain has the shape of a shape vector ({@code t.shape}, {@code v.as_list()}, or a {@code slice}
   * of one), without resolving any shapes or bounds. Used to decide whether a shape operand should
   * be left to the generator-side provenance walk rather than pinned (wala/ML#703).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves helper calls
   *     (wala/ML#706).
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to test.
   * @return {@code true} iff the def-use chain matches a shape-vector form.
   */
  public static boolean isShapeVectorChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    return isShapeVectorChain(builder, node, vn, HashSetFactory.make());
  }

  /**
   * Core of {@link #isShapeVectorChain(PropagationCallGraphBuilder, CGNode, int)} threading the set
   * of callee nodes already on the walk, guarding recursive helpers (wala/ML#706).
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves helper calls.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to test.
   * @param visited The callee nodes already on the walk.
   * @return {@code true} iff the def-use chain matches a shape-vector form.
   */
  private static boolean isShapeVectorChain(
      PropagationCallGraphBuilder builder, CGNode node, int vn, Set<CGNode> visited) {
    if (vn <= 0 || node.getDU() == null || node.getIR() == null) return false;
    SSAInstruction def = node.getDU().getDef(vn);
    SymbolTable st = node.getIR().getSymbolTable();

    if (def instanceof PythonPropertyRead) {
      int memberVn = ((PythonPropertyRead) def).getMemberRef();
      return st.isStringConstant(memberVn) && "shape".equals(st.getStringValue(memberVn));
    }

    if (def instanceof PythonInvokeInstruction) {
      PythonInvokeInstruction invoke = (PythonInvokeInstruction) def;
      if (invoke.getNumberOfUses() < 1) return false;
      SSAInstruction funcDef = node.getDU().getDef(invoke.getUse(0));
      if (funcDef instanceof PythonPropertyRead) {
        int memberVn = ((PythonPropertyRead) funcDef).getMemberRef();
        return st.isStringConstant(memberVn)
            && "as_list".equals(st.getStringValue(memberVn))
            && isShapeVectorChain(
                builder, node, ((PythonPropertyRead) funcDef).getObjectRef(), visited);
      }
      if (dispatchesToSliceBuiltin(builder, node, invoke) && invoke.getNumberOfUses() >= 2)
        return isShapeVectorChain(builder, node, invoke.getUse(1), visited);

      // A call to a user helper: the chain is a shape vector iff every callee's every returned
      // value is (wala/ML#706).
      Set<CGNode> targets = builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
      if (targets == null || targets.isEmpty()) return false;
      for (CGNode callee : targets) {
        if (callee.getIR() == null || callee.getDU() == null) return false;
        if (!visited.add(callee)) return false; // Recursive helper: unmodeled.
        try {
          boolean sawReturn = false;
          for (Iterator<SSAInstruction> it = callee.getIR().iterateAllInstructions();
              it.hasNext(); ) {
            SSAInstruction instruction = it.next();
            if (!(instruction instanceof SSAReturnInstruction)) continue;
            SSAReturnInstruction ret = (SSAReturnInstruction) instruction;
            // A bare `return` means the helper can produce None on some path, so the call isn't
            // a shape-vector chain.
            if (ret.getResult() < 0) return false;
            sawReturn = true;
            if (!isShapeVectorChain(builder, callee, ret.getResult(), visited)) return false;
          }
          if (!sawReturn) return false;
        } finally {
          visited.remove(callee);
        }
      }
      return true;
    }

    // a + b: a concatenation is a shape-vector chain iff each operand is one or is a literal
    // list construction (wala/ML#708).
    if (def instanceof SSABinaryOpInstruction
        && ((SSABinaryOpInstruction) def).getOperator() == IBinaryOpInstruction.Operator.ADD) {
      boolean leftChain = isShapeVectorChain(builder, node, def.getUse(0), visited);
      boolean rightChain = isShapeVectorChain(builder, node, def.getUse(1), visited);
      if (!leftChain && !rightChain) return false;
      for (int u = 0; u < 2; u++) {
        if (u == 0 ? leftChain : rightChain) continue;
        SSAInstruction operandDef = node.getDU().getDef(def.getUse(u));
        if (!(operandDef instanceof SSANewInstruction)) return false;
        TypeReference allocated = ((SSANewInstruction) operandDef).getNewSite().getDeclaredType();
        if (!allocated.equals(list) && !allocated.equals(tuple)) return false;
      }
      return true;
    }
    return false;
  }

  /**
   * Sentinel returned by {@link #sliceBoundOrNull} for a bound that is present but not a
   * compile-time constant. Compared by value ({@link Objects#equals}), so a genuine {@link
   * Integer#MIN_VALUE} bound in user code conservatively degrades to ⊤ along with actually
   * unresolved bounds.
   */
  protected static final Integer UNRESOLVED_BOUND = Integer.MIN_VALUE;

  /**
   * Resolves a slice bound at the given use index to a constant integer, {@code null} for an
   * absent/{@code None} bound, or {@link #UNRESOLVED_BOUND} when the bound exists but isn't a
   * compile-time constant. Reads the symbol table first (literal bounds), then a single {@code
   * ConstantKey} in the bound's points-to set (propagated constants).
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param node The {@link CGNode} whose IR contains the invoke.
   * @param st The node's symbol table.
   * @param invoke The slice-builtin invoke.
   * @param useIndex The use index of the bound ({@code 2}=lower, {@code 3}=upper, {@code 4}=step).
   * @return The constant bound, {@code null} for {@code None}/absent, or {@link #UNRESOLVED_BOUND}.
   */
  private static Integer sliceBoundOrNull(
      PropagationCallGraphBuilder builder,
      CGNode node,
      SymbolTable st,
      PythonInvokeInstruction invoke,
      int useIndex) {
    if (useIndex >= invoke.getNumberOfUses()) return null; // Absent: Python default.
    return constantIntValueOrSentinel(builder, node, st, invoke.getUse(useIndex));
  }

  /**
   * Resolves a value number to a constant integer, {@code null} for {@code None}, or {@link
   * #UNRESOLVED_BOUND}. Reads the symbol table first (literals), then constant-folds a unary
   * negation of a resolvable operand ({@code -k} for a constant {@code k}, the {@code
   * shape[-num_inner_dims:]} idiom — the PA does not fold arithmetic, so the negated value itself
   * has an empty points-to set), then falls back to a single {@code ConstantKey} in the value's
   * points-to set (propagated constants).
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param st The node's symbol table.
   * @param vn The value number to resolve.
   * @return The constant value, {@code null} for {@code None}, or {@link #UNRESOLVED_BOUND}.
   */
  private static Integer constantIntValueOrSentinel(
      PropagationCallGraphBuilder builder, CGNode node, SymbolTable st, int vn) {
    if (vn <= 0) return null;
    if (st.isNullConstant(vn)) return null; // Explicit None.
    if (st.isNumberConstant(vn)) return ((Number) st.getConstantValue(vn)).intValue();

    // Constant-fold a unary minus of a resolvable operand.
    SSAInstruction def = node.getDU() != null ? node.getDU().getDef(vn) : null;
    if (def instanceof SSAUnaryOpInstruction
        && CAstUnaryOp.MINUS.equals(((SSAUnaryOpInstruction) def).getOpcode())) {
      Integer operand = constantIntValueOrSentinel(builder, node, st, def.getUse(0));
      if (operand == null || Objects.equals(operand, UNRESOLVED_BOUND)) return UNRESOLVED_BOUND;
      return -operand;
    }

    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    if (pk == null) return UNRESOLVED_BOUND;
    OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
    Integer found = null;
    boolean sawNone = false;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return UNRESOLVED_BOUND;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (value == null) { // Propagated None.
        sawNone = true;
        continue;
      }
      if (!(value instanceof Number)) return UNRESOLVED_BOUND;
      int intValue = ((Number) value).intValue();
      if (found != null && found != intValue) return UNRESOLVED_BOUND; // Ambiguous.
      found = intValue;
    }
    // A points-to set holding both None and a numeric constant is ambiguous; collapsing it to
    // either would assert a bound the other execution violates.
    if (sawNone) return found == null ? null : UNRESOLVED_BOUND;
    return found == null ? UNRESOLVED_BOUND : found;
  }

  /**
   * Returns whether the given value number is defined by a {@code slice(...)} object construction
   * (an invoke whose function object is a {@link PythonTypes#SLICE_BUILTIN} allocation).
   *
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The value number to test.
   * @return {@code true} iff {@code vn} is defined by a slice-object construction.
   */
  private static boolean isSliceObjectDef(CGNode node, int vn) {
    if (vn <= 0) return false;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof SSAAbstractInvokeInstruction)) return false;
    SSAAbstractInvokeInstruction inv = (SSAAbstractInvokeInstruction) def;
    if (inv.getNumberOfUses() < 1) return false;
    SSAInstruction funcDef = node.getDU().getDef(inv.getUse(0));
    return funcDef instanceof SSANewInstruction
        && ((SSANewInstruction) funcDef)
            .getNewSite()
            .getDeclaredType()
            .equals(PythonTypes.SLICE_BUILTIN);
  }

  /**
   * Dtype twin of {@link #getShapeFromShapeAttributeArgument(PropagationCallGraphBuilder, int,
   * String)}: recovers an allocator's dtype when its dtype argument is another tensor's {@code
   * .dtype} (e.g. {@code tf.ones((2, 1), dtype=y.dtype)}). Such an argument is a {@link
   * PythonPropertyRead} of member {@code "dtype"} whose points-to set is empty, so ordinary dtype
   * resolution falls through to {@link #getDefaultDTypes}; this resolves the dtype of the
   * underlying tensor instead of taking the allocator's default (<a
   * href="https://github.com/wala/ML/issues/686">wala/ML#686</a>).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param paramPos The 0-based positional index of the dtype parameter (excluding {@code self}).
   * @param paramName The dtype parameter's keyword name, or {@code null}.
   * @return The union of the source tensors' dtypes, or {@code null} if no {@code .dtype} argument
   *     is found or none resolves.
   */
  protected Set<DType> getDTypeFromDTypeAttributeArgument(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    Set<DType> combined = null;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn <= 0) continue;
      SSAInstruction def = caller.getDU().getDef(argVn);
      if (!(def instanceof PythonPropertyRead)) continue;
      PythonPropertyRead propRead = (PythonPropertyRead) def;
      SymbolTable st = caller.getIR().getSymbolTable();
      int memberVn = propRead.getMemberRef();
      if (!st.isStringConstant(memberVn) || !"dtype".equals(st.getStringValue(memberVn))) continue;
      int tensorVn = propRead.getObjectRef();
      try {
        Set<DType> tensorDTypes = this.getDTypes(builder, caller, tensorVn);
        if (tensorDTypes != null && !tensorDTypes.isEmpty()) {
          if (combined == null) combined = EnumSet.noneOf(DType.class);
          combined.addAll(tensorDTypes);
        }
      } catch (IllegalArgumentException e) {
        // The source tensor's dtype couldn't be resolved; fall through to the default.
        LOGGER.log(
            Level.FINE,
            "getDTypeFromDTypeAttributeArgument: could not resolve .dtype source tensorVn="
                + tensorVn
                + " in caller="
                + caller,
            e);
      }
    }
    return combined;
  }

  /**
   * Dtype counterpart of {@link #getArgumentShapesViaCallers(PropagationCallGraphBuilder, int,
   * String)}. See that method for the rationale; the behaviour is the same but returns a set of
   * {@link DType}s resolved via {@link #getDTypes(PropagationCallGraphBuilder, CGNode, int)}.
   */
  protected Set<DType> getArgumentDTypesViaCallers(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    Set<DType> combined = null;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int argVn = -1;
      if (paramName != null) argVn = pyCall.getUse(paramName);
      if (argVn == -1 && paramPos >= 0) {
        int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
        if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
      }
      if (argVn <= 0) continue;
      try {
        Set<DType> argDTypes = this.getDTypes(builder, caller, argVn);
        if (argDTypes != null && !argDTypes.isEmpty()) {
          if (combined == null) combined = EnumSet.noneOf(DType.class);
          combined.addAll(argDTypes);
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(Level.FINE, "Could not get dtypes for caller argument: " + argVn, e);
      }
    }
    return combined;
  }

  /**
   * Retrieves the points-to set for a specific argument of the function call represented by this
   * generator (or specifically for the given node).
   *
   * <p>This method employs a multi-staged strategy to resolve arguments:
   *
   * <ol>
   *   <li><b>Direct Invoke Instruction:</b> If the invoke instruction is directly available (via
   *       {@link #getInvokeInstruction()}), it resolves the argument using the parameter name (for
   *       keyword arguments) or position.
   *   <li><b>Direct Binary Op Instruction:</b> If the instruction is a binary operation (via {@link
   *       #getBinaryOpInstruction()}), it resolves the argument using the position (0 for left, 1
   *       for right).
   *   <li><b>`read_data` Wrapper Handling:</b> If the node corresponds to a `read_data` method
   *       (common in TensorFlow synthetic models), it delegates the resolution to the preceding
   *       `do` method, which is the actual entry point for the operation logic.
   *   <li><b>Caller Analysis:</b> It walks the call-graph edges to the calling invocations to
   *       identify the specific caller of the node. It then inspects the call sites in that caller
   *       to find the {@link PythonInvokeInstruction} that targets this node. This is crucial for
   *       distinguishing between different calls to the same operation in a context-sensitive
   *       manner.
   *   <li><b>Callee Parameter Fallback:</b> If the argument cannot be resolved from the caller
   *       (e.g., due to analysis imprecision or manual node creation), it attempts to resolve it
   *       directly from the parameter value numbers within the callee node itself.
   * </ol>
   *
   * @param builder the propagation call graph builder
   * @param node the call graph node representing the function execution
   * @param paramPos the 0-based index of the positional parameter (excluding 'self' for instance
   *     methods)
   * @param paramName the name of the keyword parameter
   * @return the points-to set of the argument, or {@link OrdinalSet#empty()} if not found
   */
  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, CGNode node, int paramPos, String paramName) {
    // Strategy 1: Direct Invoke Instruction
    // If we have direct access to the PythonInvokeInstruction, use it. This is the most direct and
    // reliable method
    // when the generator is instantiated with a source that directly points to the result of an
    // invoke.
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int argValNum = -1;

      // Try to resolve by name (keyword argument) first.
      if (paramName != null) {
        argValNum = call.getUse(paramName);
      }

      // If not found by name, try by position.
      if (argValNum == -1) {
        if (paramPos == RECEIVER_PARAMETER_POSITION) {
          argValNum = getReceiverValueNumber(node, call);
        } else if (paramPos >= 0) {
          // Adjust position to account for keyword arguments which are stored at the end of the use
          // list.
          int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
          // Total uses minus the function object itself (index 0) and the keyword args.
          int numPosParams = call.getNumberOfUses() - 1 - numKeywords;

          if (paramPos < numPosParams) {
            // Positional arguments start at index 1 (index 0 is the function object).
            argValNum = call.getUse(paramPos + 1);
          }
        }
      }

      if (argValNum > 0) {
        PointerKey argPk =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, argValNum);
        OrdinalSet<InstanceKey> argPts = builder.getPointerAnalysis().getPointsToSet(argPk);
        if (argPts != null && !argPts.isEmpty()) {
          return argPts;
        }
      }
      return OrdinalSet.empty();
    }

    // Strategy 1.5: Direct Binary Op Instruction
    SSABinaryOpInstruction binOp = getBinaryOpInstruction();
    if (binOp != null) {
      int argValNum = -1;
      if (paramPos >= 0 && paramPos < binOp.getNumberOfUses()) {
        argValNum = binOp.getUse(paramPos);
      }
      if (argValNum > 0) {
        PointerKey argPk =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, argValNum);
        return builder.getPointerAnalysis().getPointsToSet(argPk);
      }
      return OrdinalSet.empty();
    }

    // Strategy 2: `read_data` Wrapper Handling
    // Synthetic TensorFlow models often use a `read_data` helper method. If we are analyzing such a
    // node,
    // we need to step back to the caller (usually the `do` method of the operation) to find the
    // actual arguments.
    if (node.getMethod().getName().toString().equals(TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME)) {
      OrdinalSet<InstanceKey> ret = OrdinalSet.empty();
      Iterator<CGNode> preds = builder.getCallGraph().getPredNodes(node);
      while (preds.hasNext()) {
        CGNode pred = preds.next();
        if (pred.getMethod().getName().toString().equals(DO_METHOD_NAME)) {
          ret = OrdinalSet.unify(ret, getArgumentPointsToSet(builder, pred, paramPos, paramName));
        }
      }
      return ret;
    }

    // Strategy 3: Caller Analysis
    // Walk the call-graph edges to the invocations dispatching to this node, then look up the
    // arguments passed at those call sites in each caller's IR.
    {
      OrdinalSet<InstanceKey> combinedPts = OrdinalSet.empty();
      boolean found = false;
      boolean callAnalyzed = false;

      for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
          getCallerInvokes(builder, node)) {
        CGNode caller = callerInvoke.fst;
        SSAAbstractInvokeInstruction callInstr = callerInvoke.snd;
        callAnalyzed = true;

        if (callInstr instanceof PythonInvokeInstruction) {
          PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
          int argValNum = -1;

          // Try to resolve by name (keyword argument).
          if (paramName != null) {
            argValNum = pyCallInstr.getUse(paramName);
          }

          // Try to resolve by position.
          if (argValNum == -1) {
            if (paramPos == RECEIVER_PARAMETER_POSITION) {
              argValNum = getReceiverValueNumber(caller, pyCallInstr);
            } else if (paramPos >= 0) {
              int numPosParams =
                  pyCallInstr.getNumberOfPositionalParameters() - 1; // Exclude function.
              if (paramPos < numPosParams) {
                argValNum =
                    pyCallInstr.getUse(paramPos + 1); // Positional arguments start at index 1.
              }
            }
          }

          if (argValNum > 0) {
            PointerKey argPk =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(caller, argValNum);
            OrdinalSet<InstanceKey> argPts = builder.getPointerAnalysis().getPointsToSet(argPk);
            if (argPts != null && !argPts.isEmpty()) {
              combinedPts = OrdinalSet.unify(combinedPts, argPts);
              found = true;
            }
          }
        }
      }

      if (found) {
        return combinedPts;
      }
      // If we analyzed the call but couldn't find the argument, it likely wasn't provided.
      if (callAnalyzed) {
        return OrdinalSet.empty();
      }
    }

    // Strategy 4: Callee Parameter Fallback
    // If we couldn't resolve the argument from the caller, we look at the parameter value numbers
    // within the callee (the `node` itself). This assumes the argument was successfully passed
    // and mapped to a local variable in the callee.
    int valNum = this.getArgumentValueNumber(node, paramPos);
    if (valNum > 0) {
      PointerKey pk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valNum);
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
      if (pts != null && !pts.isEmpty()) {
        return pts;
      }
    }

    return OrdinalSet.empty();
  }

  private int getArgumentValueNumber(CGNode node, int parameterPosition) {
    if (parameterPosition == RECEIVER_PARAMETER_POSITION) return node.getIR().getParameter(0);
    if (parameterPosition < 0) return UNDEFINED_PARAMETER_POSITION; // No such argument.

    int index = node.getMethod().isStatic() ? parameterPosition : parameterPosition + 1;

    if (index >= node.getIR().getNumberOfParameters()) return UNDEFINED_PARAMETER_POSITION;

    return node.getIR().getParameter(index);
  }

  /**
   * Returns the value number for the argument at the specified position or with the specified name.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @param paramName The name of the argument in the function call.
   * @param optional Whether the argument is optional.
   * @return The value number for the argument at the specified position or with the specified name
   *     or -1 if the argument is optional and not present.
   * @throws IllegalStateException If the argument is mandatory and not present.
   */
  protected int getArgumentValueNumber(
      PropagationCallGraphBuilder builder, int paramPos, String paramName, boolean optional) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (this.getNode()
        .getMethod()
        .getName()
        .toString()
        .equals(TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME)) {
      // For read_data nodes, we don't have explicit arguments in the IR.
      // Returning MAX_VALUE acts as a sentinel to bypass the "missing argument" check below
      // and allows getDTypes/getShapes to proceed to getArgumentPointsToSet,
      // which correctly delegates argument resolution to the caller (do).
      return Integer.MAX_VALUE;
    }

    if (call != null) {
      int argValNum = -1;

      if (paramName != null) {
        argValNum = call.getUse(paramName);
      }

      if (argValNum == -1 && paramPos >= 0) {
        int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
        int numPosParams =
            call.getNumberOfUses() - 1 - numKeywords; // Exclude function and keywords.

        if (paramPos < numPosParams) {
          argValNum = call.getUse(paramPos + 1); // Positional arguments start at index 1.
        }
      }

      if (argValNum != -1) return argValNum;

      if (optional) return -1;
      else
        throw new IllegalStateException(
            "Cannot determine value number for parameter at position "
                + paramPos
                + (paramName == null ? "" : " or name " + paramName)
                + " of "
                + this.getSignature());
    }

    SSABinaryOpInstruction binOp = getBinaryOpInstruction();
    if (binOp != null) {
      if (paramPos >= 0 && paramPos < binOp.getNumberOfUses()) {
        return binOp.getUse(paramPos);
      }
      if (optional) return -1;
      throw new IllegalStateException(
          "Cannot determine value number for binary op parameter at position " + paramPos);
    } else {
      // Fallback for manual nodes (no invoke instruction).
      // We assume the arguments are available as parameters in the method body.
      if (paramPos >= 0) {
        return this.getArgumentValueNumber(this.getNode(), paramPos);
      }
    }

    Set<Integer> numArgs = this.getNumberOfPossiblePositionalArguments(builder);

    boolean keywordPresent =
        (paramName != null && this.isKeywordArgumentPresent(builder, paramName));
    boolean positionalPresent = numArgs.stream().anyMatch(n -> n > paramPos);

    if (!positionalPresent && !keywordPresent)
      if (optional) return -1;
      else
        throw new IllegalStateException(
            "Cannot determine value number for parameter at position "
                + paramPos
                + (paramName == null ? "" : " or name " + paramName)
                + " of "
                + this.getSignature());

    return this.getArgumentValueNumber(paramPos);
  }

  /**
   * Returns whether the keyword argument with the specified name is present in the function call.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramName The name of the keyword argument.
   * @return {@code true} if the keyword argument is present, {@code false} otherwise.
   */
  protected boolean isKeywordArgumentPresent(
      PropagationCallGraphBuilder builder, String paramName) {
    // 1. Try to resolve the call directly from the definition of the value.
    // This works if we are analyzing the code where the function was called (the caller).
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      return call.getKeywords().contains(paramName);
    }

    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      if (callerInvoke.snd instanceof PythonInvokeInstruction) {
        PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callerInvoke.snd;
        if (pyCallInstr.getKeywords().contains(paramName)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Returns the value number for the argument at the specified position.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @param optional Whether the argument is optional.
   * @return The value number for the argument at the specified position or -1 if the argument is
   *     optional and not present.
   * @throws IllegalStateException If the argument is mandatory and not present.
   */
  protected int getArgumentValueNumber(
      PropagationCallGraphBuilder builder, int paramPos, boolean optional) {
    return this.getArgumentValueNumber(builder, paramPos, null, optional);
  }

  /**
   * Returns the value number for the argument at the specified position.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @return The value number for the argument at the specified position.
   * @throws IllegalStateException If the argument is mandatory and not present.
   */
  protected int getArgumentValueNumber(PropagationCallGraphBuilder builder, int paramPos) {
    return this.getArgumentValueNumber(builder, paramPos, false);
  }

  /**
   * Returns the set of possible numbers of positional arguments passed to the range function at the
   * call.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @return A set of integers representing the possible number of positional arguments.
   */
  protected Set<Integer> getNumberOfPossiblePositionalArguments(
      PropagationCallGraphBuilder builder) {
    Set<Integer> ret = HashSetFactory.make();

    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
      ret.add(call.getNumberOfUses() - 1 - numKeywords);
      return ret;
    }

    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      SSAAbstractInvokeInstruction callInstr = callerInvoke.snd;
      LOGGER.finest(() -> "Call instruction: " + callInstr + ".");

      if (callInstr instanceof PythonInvokeInstruction) {
        PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
        int numKeywords = pyCallInstr.getKeywords() != null ? pyCallInstr.getKeywords().size() : 0;
        int numberOfPositionalParameters =
            pyCallInstr.getNumberOfUses() - 1 - numKeywords; // Exclude the function name and
        // keywords.

        ret.add(numberOfPositionalParameters);
      }
    }

    return ret;
  }

  /**
   * Returns the possible double values for the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param caller The {@link CGNode} calling the function.
   * @param vn The value number of the argument.
   * @return A set of possible double values, or {@code null} if the argument's points-to set
   *     contains a non-constant key (the value is not statically resolvable, wala/ML#669).
   */
  protected static Set<Double> getPossibleDoubleValues(
      PropagationCallGraphBuilder builder, CGNode caller, int vn) {
    Set<Double> vals = HashSetFactory.make();
    if (vn == -1) return vals;

    // 1. Try symbol table (for literal constants)
    if (caller.getIR().getSymbolTable().isConstant(vn)) {
      Object val = caller.getIR().getSymbolTable().getConstantValue(vn);
      if (val instanceof Number) {
        vals.add(((Number) val).doubleValue());
      }
    }

    // 2. Try points-to analysis
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, vn);
    Set<Double> fromPts = getPossibleDoubleValues(builder.getPointerAnalysis().getPointsToSet(pk));
    // A non-constant key means the value is not statically resolvable (wala/ML#669); a partial
    // result would falsely claim exhaustiveness.
    if (fromPts == null) return null;
    vals.addAll(fromPts);

    return vals;
  }

  /**
   * Returns the possible double values for the given points-to set.
   *
   * @param pts The points-to set of the argument.
   * @return A set of possible double values, or {@code null} if the points-to set contains a
   *     non-constant key (the value is not statically resolvable, wala/ML#669).
   */
  protected static Set<Double> getPossibleDoubleValues(OrdinalSet<InstanceKey> pts) {
    Set<Object> constants = getConstantValues(pts, true);
    if (constants == null) return null;

    Set<Double> ret = HashSetFactory.make();

    for (Object val : constants) {
      if (val instanceof Number) {
        ret.add(((Number) val).doubleValue());
      } else if (val == null) {
        ret.add(null);
      } else {
        throw new IllegalStateException("Expected a number but found: " + val.getClass() + ".");
      }
    }

    return ret;
  }

  /**
   * Returns the possible long values for the given points-to set. If the value is `None`, then a
   * null value will be contained within the returned set.
   *
   * @param pointsToSet The points-to set of the value.
   * @return A set of possible long values, or {@code null} if the points-to set contains a
   *     non-constant key (the value is not statically resolvable, wala/ML#669). If the value is
   *     `None`, then a null value will be contained within the returned set.
   */
  protected static Set<Long> getPossibleLongValues(OrdinalSet<InstanceKey> pointsToSet) {
    Set<Object> constants = getConstantValues(pointsToSet, true);
    if (constants == null) return null;

    Set<Long> ret = HashSetFactory.make();

    for (Object val : constants) {
      if (val instanceof Number) {
        ret.add(((Number) val).longValue());
      } else if (val == null) {
        ret.add(null);
      } else {
        throw new IllegalStateException("Expected a number but found: " + val.getClass() + ".");
      }
    }

    return ret;
  }

  /**
   * Returns a set of constant values derived from the given points-to set.
   *
   * @param pts The points-to set to analyze.
   * @param requireConstants If true, a non-constant key makes the whole set non-static: {@code
   *     null} is returned instead of the partial constants. If false, non-constant keys are
   *     skipped.
   * @return A set of constant values (which may contain nulls), or {@code null} if {@code
   *     requireConstants} is true and a non-constant key is found (the value is not statically
   *     resolvable). Previously this case threw {@link IllegalStateException}, which aborted the
   *     whole analysis when WALA 1.8.0 supplied a {@code ScopeMappingInstanceKey} (<a
   *     href="https://github.com/wala/ML/issues/669">wala/ML#669</a>).
   */
  protected static Set<Object> getConstantValues(
      OrdinalSet<InstanceKey> pts, boolean requireConstants) {
    Set<Object> ret = HashSetFactory.make();

    if (pts != null) {
      for (InstanceKey ik : pts) {
        if (ik instanceof ConstantKey) {
          ret.add(((ConstantKey<?>) ik).getValue());
        } else if (requireConstants) {
          LOGGER.fine(
              () ->
                  "Non-constant key: "
                      + ik.getClass()
                      + " in points-to set; the value is not statically resolvable (wala/ML#669).");
          return null;
        }
      }
    }

    return ret;
  }

  /**
   * Creates a manual TensorGenerator for a specific allocation site within a synthetic method.
   *
   * <p>This overload dispatches on the concrete type of the allocation rather than on the declaring
   * class of the containing method. It is needed when a single synthetic {@code do()} body
   * allocates multiple objects of distinct types — e.g., {@code load_data/do()} allocates both
   * {@code x_train} and {@code y_train} with different shapes.
   *
   * @param node The {@link CGNode} containing the allocation.
   * @param allocationType The concrete {@link TypeReference} of the allocated object.
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A {@link TensorGenerator} for the allocation, or {@code null} if the type is not
   *     recognized.
   */
  protected static TensorGenerator createManualGenerator(
      CGNode node, TypeReference allocationType, PropagationCallGraphBuilder builder) {
    LOGGER.fine("createManualGenerator checking allocation type: " + allocationType.getName());

    TypeReference sanitized = sanitize(allocationType);

    if (sanitized.equals(TensorFlowTypes.MNIST_X_TRAIN)) {
      return new MnistInputData(node, MnistInputData.X_TRAIN_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.MNIST_Y_TRAIN)) {
      return new MnistInputData(node, MnistInputData.Y_TRAIN_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.MNIST_X_TEST)) {
      return new MnistInputData(node, MnistInputData.X_TEST_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.MNIST_Y_TEST)) {
      return new MnistInputData(node, MnistInputData.Y_TEST_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.CIFAR10_X_TRAIN)) {
      return new Cifar10InputData(node, Cifar10InputData.X_TRAIN_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.CIFAR10_Y_TRAIN)) {
      return new Cifar10InputData(node, Cifar10InputData.Y_TRAIN_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.CIFAR10_X_TEST)) {
      return new Cifar10InputData(node, Cifar10InputData.X_TEST_SHAPE);
    } else if (sanitized.equals(TensorFlowTypes.CIFAR10_Y_TEST)) {
      return new Cifar10InputData(node, Cifar10InputData.Y_TEST_SHAPE);
    }

    return createManualGenerator(node, builder);
  }

  /**
   * Creates a manual TensorGenerator for synthetic operations where standard points-to analysis
   * fails (e.g. UnimplementedError due to implicit pointer keys for allocations in synthetic do()
   * methods).
   */
  protected static TensorGenerator createManualGenerator(
      CGNode node, PropagationCallGraphBuilder builder) {
    TypeReference type = node.getMethod().getDeclaringClass().getReference();
    LOGGER.fine("createManualGenerator checking type: " + type.getName());

    // sanitize the type name by removing the artificial suffix that is added for synthetic
    // classes to facilitate trampoline generation.
    type = sanitize(type);

    LOGGER.fine("createManualGenerator checking sanitized type: " + type.getName());

    if (type.equals(TensorFlowTypes.SPLIT.getDeclaringClass())) {
      return new Split(node);
    } else if (type.equals(TensorFlowTypes.ONES.getDeclaringClass())) {
      return new Ones(node);
    } else if (type.equals(TensorFlowTypes.ZEROS.getDeclaringClass())) {
      return new Zeros(node);
    } else if (type.equals(NumpyTypes.ONES.getDeclaringClass())) {
      return new NpOnes(node);
    } else if (type.equals(NumpyTypes.ZEROS.getDeclaringClass())) {
      return new NpZeros(node);
    } else if (type.equals(TensorFlowTypes.SPARSE_EYE.getDeclaringClass())) {
      return new SparseEye(node);
    } else if (type.equals(TensorFlowTypes.EYE.getDeclaringClass())) {
      return new Eye(node);
    } else if (type.equals(TensorFlowTypes.UNIFORM.getDeclaringClass())) {
      return new Uniform(node);
    } else if (type.equals(TensorFlowTypes.NORMAL.getDeclaringClass())) {
      return new Normal(node);
    } else if (type.equals(TensorFlowTypes.TRUNCATED_NORMAL.getDeclaringClass())) {
      return new TruncatedNormal(node);
    } else if (type.equals(TensorFlowTypes.GAMMA.getDeclaringClass())) {
      return new Gamma(node);
    } else if (type.equals(TensorFlowTypes.POISSON.getDeclaringClass())) {
      return new Poisson(node);
    } else if (type.equals(TensorFlowTypes.VARIABLE.getDeclaringClass())) {
      return new Variable(node);
    } else if (type.equals(TensorFlowTypes.ADD_WEIGHT.getDeclaringClass())) {
      return new AddWeight(node);
    } else if (type.equals(TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE)
        || type.getName().toString().equals("Ltensorflow/data/from_tensor_slices")) {
      return new DatasetFromTensorSlicesGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_FROM_TENSORS_TYPE)
        || type.getName().toString().equals("Ltensorflow/data/from_tensors")) {
      return new DatasetFromTensorsGenerator(node);
    } else if (type.equals(DATASET_CHOOSE_FROM_DATASETS_TYPE)) {
      return new DatasetChooseFromDatasetsGenerator(node);
    } else if (type.equals(DATASET_SAMPLE_FROM_DATASETS_TYPE)) {
      return new DatasetSampleFromDatasetsGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_FROM_GENERATOR_TYPE)) {
      return new DatasetFromGeneratorGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_ZIP_TYPE)) {
      return new DatasetZipGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_RANGE_TYPE)) {
      return new DatasetRangeGenerator(node);
    } else if (type.equals(TensorFlowTypes.TEXT_LINE_DATASET_TYPE)) {
      return new TextLineDatasetGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_RANDOM_TYPE)) {
      return new DatasetRandomGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_BATCH_TYPE)) {
      return new DatasetBatchGenerator(node);
    } else if (type.equals(TensorFlowTypes.IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE)) {
      return new FlowFromDirectoryGenerator(node);
    } else if (type.equals(TensorFlowTypes.MNIST_X_TRAIN)) {
      return new MnistInputData(node, MnistInputData.X_TRAIN_SHAPE);
    } else if (type.equals(TensorFlowTypes.MNIST_Y_TRAIN)) {
      return new MnistInputData(node, MnistInputData.Y_TRAIN_SHAPE);
    } else if (type.equals(TensorFlowTypes.MNIST_X_TEST)) {
      return new MnistInputData(node, MnistInputData.X_TEST_SHAPE);
    } else if (type.equals(TensorFlowTypes.MNIST_Y_TEST)) {
      return new MnistInputData(node, MnistInputData.Y_TEST_SHAPE);
    } else if (type.equals(TensorFlowTypes.CIFAR10_X_TRAIN)) {
      return new Cifar10InputData(node, Cifar10InputData.X_TRAIN_SHAPE);
    } else if (type.equals(TensorFlowTypes.CIFAR10_Y_TRAIN)) {
      return new Cifar10InputData(node, Cifar10InputData.Y_TRAIN_SHAPE);
    } else if (type.equals(TensorFlowTypes.CIFAR10_X_TEST)) {
      return new Cifar10InputData(node, Cifar10InputData.X_TEST_SHAPE);
    } else if (type.equals(TensorFlowTypes.CIFAR10_Y_TEST)) {
      return new Cifar10InputData(node, Cifar10InputData.Y_TEST_SHAPE);
    } else if (type.getName().toString().startsWith(TensorFlowTypes.DATA_PACKAGE_PREFIX)) {
      return new DatasetGenerator(node);
    } else if (type.equals(TensorFlowTypes.MATMUL.getDeclaringClass())) {
      return new MatMul(node);
    } else if (type.equals(TensorFlowTypes.MULTIPLY.getDeclaringClass())) {
      return new ElementWiseOperation(node);
    } else if (type.equals(TensorFlowTypes.TRANSPOSE.getDeclaringClass())) {
      return new Transpose(node);
    } else if (type.equals(TensorFlowTypes.RESHAPE.getDeclaringClass())) {
      return new Reshape(node);
    } else if (type.equals(TensorFlowTypes.SQUEEZE.getDeclaringClass())) {
      return new Squeeze(node);
    } else if (type.equals(TensorFlowTypes.CONCAT.getDeclaringClass())) {
      return new Concat(node);
    } else if (type.equals(TensorFlowTypes.FILL.getDeclaringClass())) {
      return new Fill(node);
    } else if (type.equals(TensorFlowTypes.ONE_HOT.getDeclaringClass())) {
      return new OneHot(node);
    } else if (type.equals(TensorFlowTypes.REDUCE_MEAN.getDeclaringClass())) {
      return new ReduceMean(node);
    } else if (type.equals(TensorFlowTypes.CONVERT_TO_TENSOR.getDeclaringClass())) {
      return new ConvertToTensor(node);
    } else if (type.equals(TensorFlowTypes.RANGE.getDeclaringClass())) {
      return new Range(node);
    } else if (type.equals(NumpyTypes.ARRAY.getDeclaringClass())) {
      return new NpArray(node);
    } else if (type.equals(TensorFlowTypes.UNSORTED_SEGMENT_SUM.getDeclaringClass())
        || type.equals(TensorFlowTypes.UNSORTED_SEGMENT_MAX.getDeclaringClass())
        || type.equals(TensorFlowTypes.UNSORTED_SEGMENT_MEAN.getDeclaringClass())) {
      return new UnsortedSegmentReduction(node);
    } else if (type.equals(TensorFlowTypes.SIGMOID.getDeclaringClass())) {
      return new Sigmoid(node);
    } else if (type.equals(TensorFlowTypes.SOFTMAX.getDeclaringClass())) {
      return new Softmax(node);
    } else if (type.equals(TensorFlowTypes.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass())
        || type.equals(
            TensorFlowTypes.SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass())) {
      return new SoftmaxCrossEntropy(node);
    } else if (type.equals(TensorFlowTypes.CAST.getDeclaringClass())) {
      return new Cast(node);
    } else if (type.equals(TensorFlowTypes.RELU.getDeclaringClass())) {
      return new Relu(node);
    } else if (type.equals(TensorFlowTypes.EXPAND_DIMS.getDeclaringClass())) {
      return new ExpandDims(node);
    } else if (type.equals(TensorFlowTypes.CLIP_BY_VALUE.getDeclaringClass())) {
      return new ClipByValue(node);
    } else if (type.equals(CONSTANT.getDeclaringClass())) {
      return new TensorGenerator(node) {
        @Override
        public String toString() {
          return "Manual Constant Generator";
        }

        @Override
        protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
          // If dtype is not provided, infer from the value (arg 0).
          return this.getDTypes(builder, this.getArgumentValueNumber(0));
        }

        @Override
        protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
          // If the shape argument is not provided, we infer the shape from the value (arg 0).
          return this.getShapes(builder, this.getArgumentValueNumber(0));
        }

        @Override
        protected int getShapeParameterPosition() {
          return UNDEFINED_PARAMETER_POSITION;
        }

        @Override
        protected String getShapeParameterName() {
          return "shape";
        }

        @Override
        protected int getDTypeParameterPosition() {
          return 1;
        }

        @Override
        protected String getDTypeParameterName() {
          return "dtype";
        }
      };
    } else if (type.equals(PythonTypes.SLICE_BUILTIN)) {
      return new SliceBuiltinOperation(node);
    } else if (type.equals(PLACEHOLDER.getDeclaringClass())) {
      return new Placeholder(node);
    } else if (type.equals(TensorFlowTypes.DENSE_CALL.getDeclaringClass())) {
      return new DenseCall(node);
    } else if (type.equals(TensorFlowTypes.GLOBAL_AVERAGE_POOLING_1D_CALL.getDeclaringClass())) {
      return new GlobalAveragePooling1DCall(node);
    } else if (type.equals(TensorFlowTypes.EMBEDDING_LAYER_CALL.getDeclaringClass())) {
      return new EmbeddingCall(node);
    } else if (type.equals(TensorFlowTypes.MODEL_CALL.getDeclaringClass())) {
      return new ModelCall(node);
    } else if (type.equals(TensorFlowTypes.MODEL.getDeclaringClass())) {
      return new Model(node);
    } else if (type.equals(TensorFlowTypes.INPUT.getDeclaringClass())) {
      return new Input(node);
    } else if (type.equals(TensorFlowTypes.VAR_LEN_FEATURE.getDeclaringClass())) {
      // The SparseTensor a `tf.io.VarLenFeature` parses to is allocated in `VarLenFeature.do()`.
      // When
      // it reaches a consumer (e.g. `tf.sparse.to_dense`) through a feature dict, the points-to
      // walk
      // lands here on the allocation site rather than the `tf.io.VarLenFeature` call, so dispatch
      // the
      // generator that reads the feature's `dtype` and emits the contract shape. wala/ML#646.
      return new VarLenFeature(node);
    } else if (type.equals(TensorFlowTypes.FIXED_LEN_FEATURE.getDeclaringClass())) {
      // The dense tensor a `tf.io.FixedLenFeature` parses to is allocated in
      // `FixedLenFeature.do()`.
      // When it reaches a consumer through a feature dict (e.g. a `parse_single_example` result
      // subscripted by feature name), the points-to walk lands here on the allocation site rather
      // than the `tf.io.FixedLenFeature` call, so dispatch the generator that reads the feature's
      // `dims` (shape) and `type` (dtype). wala/ML#655.
      return new FixedLenFeature(node);
    }
    // Unregistered type: the manual-walker dispatch table doesn't know about this op, so the
    // caller will treat the null return as "no contribution" and silently lose precision on
    // anything reached through this node. The factory-side dispatch
    // (`TensorGeneratorFactory.getGeneratorBody`) may still cover this type — the two tables
    // aren't yet unified (wala/ML#469); audit both when this fires for an op we expect to model.
    // Logged at WARNING (deduplicated per type, see `WARNED_UNREGISTERED_MANUAL_TYPES`) per
    // wala/ML#468 to make the silent-skip audible without flooding logs when the same op is hit
    // many times within one analysis.
    final TypeReference unregisteredType = type;
    if (WARNED_UNREGISTERED_MANUAL_TYPES.add(unregisteredType.getName().toString())) {
      LOGGER.warning(
          () ->
              "createManualGenerator: no manual generator registered for type "
                  + unregisteredType.getName()
                  + " (first occurrence; further calls for this type are silent). Treating as"
                  + " no contribution; this is likely a wala/ML#468 / wala/ML#469 gap in the"
                  + " manual-walker dispatch table. Add a case here if this op should be"
                  + " modeled in manual-walker contexts.");
    } else {
      LOGGER.fine(
          () ->
              "createManualGenerator: no manual generator registered for type "
                  + unregisteredType.getName()
                  + " (already warned).");
    }
    return null;
  }

  /**
   * De-duplication set for the wala/ML#468 unregistered-type WARNING in {@link
   * #createManualGenerator(CGNode, PropagationCallGraphBuilder)}. Without dedup the WARNING fires
   * per call site, which on the existing test suite produces ~1500 lines per analysis run —
   * drowning the signal. Tracking already-warned type names keeps each new gap visible exactly once
   * per JVM lifetime.
   *
   * <p>Intentionally JVM-scoped (not cleared by {@link #clearCaches}): the goal is to surface each
   * gap once per JVM run, not once per analysis. Re-warning per test still floods logs because the
   * same op gets hit by many tests.
   */
  private static final Set<String> WARNED_UNREGISTERED_MANUAL_TYPES =
      Collections.synchronizedSet(new HashSet<>());
}
