package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_TYPE;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.math.top_k(input, k=1, sorted=True, name=None)}. Returns a 2-tuple {@code
 * (values, indices)}; both elements have shape {@code input.shape[:-1] + (k,)}. {@code values} has
 * the input's dtype; {@code indices} is fixed at {@code int32}.
 *
 * <p>Implements {@link TupleElementProvider} so that destructuring or indexed accesses (e.g. {@code
 * values, indices = tf.math.top_k(x, k)} or {@code result.indices}) resolve to the right
 * per-element shape and dtype rather than collapsing both to the aggregate union. This is the first
 * non-Dataset use of the {@link TupleElementProvider} pattern; the established Dataset-side
 * precedent is {@link DatasetFromTensorsGenerator}.
 *
 * <p>The output shape {@code input.shape[:-1] + (k,)} is composed from the input tensor's shape and
 * the {@code k} argument (default {@code 1}); see {@code composedShapes} and <a
 * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>. An unresolvable {@code k} loses
 * only the last axis: with the input's rank in hand, that axis becomes a sentinel (Dynamic for a
 * tensor {@code k}, Unresolved otherwise) rather than dropping the whole shape; it degrades to ⊤
 * only when the input shape is itself unknown rank or rank-0. The per-element dtype (FLOAT32 for
 * values, INT32 for indices) is resolved independently.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/top_k">tf.math.top_k</a>
 * @see <a href="https://github.com/wala/ML/issues/449">wala/ML#449</a> (Tier 5).
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TopK extends TensorGenerator implements TupleElementProvider {

  /** Tuple-element index for the {@code values} output. */
  private static final int VALUES_INDEX = 0;

  /** Tuple-element index for the {@code indices} output (always {@code int32}). */
  private static final int INDICES_INDEX = 1;

  /**
   * Parameter positions and keyword names for {@code tf.math.top_k(input, k=1, sorted=True,
   * name=None)}. Ordinals match the position in {@code tensorflow.xml}'s {@code paramNames} after
   * the implicit {@code self} receiver.
   */
  protected enum Parameters {
    /** The input tensor; the dtype source for {@code values}. */
    INPUT,

    /** The number of top entries to return; default {@code 1}. */
    K,

    /** Whether the resulting top-k entries should be returned in sorted order. */
    SORTED,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
     *
     * @return The lowercased enum name.
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public TopK(PointsToSetVariable source) {
    super(source);
  }

  public TopK(CGNode node) {
    super(node);
  }

  /** Always yields a tuple — the call signature guarantees a {@code (values, indices)} return. */
  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    // Both values and indices share shape input.shape[:-1] + (k,). wala/ML#609.
    return this.composedShapes(builder);
  }

  /**
   * Composes the top_k output shape: {@code input.shape[:-1] + (k,)}. Resolves the input tensor's
   * shape and the {@code k} argument (default {@code 1}) and replaces the last axis with {@code k},
   * or with a sentinel ({@link #unresolvableKAxis}) when {@code k} is supplied but not a resolvable
   * constant — the rank is still known from the input. Returns ⊤ ({@code null}) only when the input
   * shape is itself unknown rank or rank-0, where there is no last axis to replace. See <a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of composed output shapes, or {@code null} (⊤) if it can't be composed.
   */
  private Set<List<Dimension<?>>> composedShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = null;
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    if (inputPts != null && !inputPts.isEmpty())
      inputShapes = this.getShapesOfValue(builder, inputPts);
    if (inputShapes == null || inputShapes.isEmpty())
      // The input is frequently overlay-resolved (an elementwise binop, a transpose) and carries no
      // points-to allocation at the synthetic top_k node, so its points-to set is empty even though
      // the analysis has its shape. Read it in the caller's frame (the wala/ML#718 caller-aware
      // path, as NpPermutation and ElementWiseOperation do). Only then is the rank genuinely
      // unknown.
      inputShapes =
          this.getArgumentShapeResultViaCallers(
                  builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName())
              .toLegacy();
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    // An unresolvable k loses only the last axis, not the rank: the output is
    // input.shape[:-1] + (k,), and the input's rank is in hand here, so replace the final extent
    // with a sentinel rather than dropping the whole shape to ⊤ (wala/ML#609 refuses to guess k's
    // VALUE; a rank-plus-wildcard guesses nothing, asserting only the rank the analysis already
    // holds). The sentinel follows wala/ML#721: DYNAMIC when k is a tensor, since TensorFlow's
    // static shape reports None for a tensor-valued k; UNRESOLVED otherwise, a fixed runtime
    // integer the analysis could not compute.
    Integer k = this.resolveK(builder);
    Dimension<?> lastAxis = (k != null) ? new NumericDim(k) : this.unresolvableKAxis(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> in : inputShapes) {
      // Unknown rank (null) or a rank-0 scalar can't have its last axis replaced; degrade to ⊤.
      if (in == null || in.isEmpty()) return null;
      List<Dimension<?>> out = new ArrayList<>(in);
      out.set(out.size() - 1, lastAxis);
      ret.add(out);
    }
    // inputShapes is non-empty and the loop returns ⊤ for any null/empty shape, so ret is
    // populated.
    return ret;
  }

  /**
   * The last-axis sentinel for an unresolvable {@code k}. {@link DynamicDim} when {@code k} is a
   * tensor — TensorFlow's static shape reports {@code None} for that axis (wala/ML#721) — and
   * {@link UnresolvedDim} otherwise, a fixed runtime integer the analysis could not compute (e.g. a
   * config value or an unmodeled Python scalar).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link DynamicDim#INSTANCE} when {@code k}'s points-to set holds a tensor allocation,
   *     else {@link UnresolvedDim#INSTANCE}.
   */
  private Dimension<?> unresolvableKAxis(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> kPts =
        this.getArgumentPointsToSet(builder, Parameters.K.getIndex(), Parameters.K.getName());
    if (kPts != null)
      for (InstanceKey instanceKey : kPts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
        if (asin != null && asin.concreteType().getReference().equals(TENSOR_TYPE))
          return DynamicDim.INSTANCE;
      }
    return UnresolvedDim.INSTANCE;
  }

  /**
   * Resolves the {@code k} argument as an integer constant. Defaults to {@code 1} only when {@code
   * k} is genuinely omitted; when {@code k} is supplied but its value can't be resolved (empty
   * points-to set, or a non-constant such as an opaque or tensor {@code k}), returns {@code null}
   * (⊤) rather than assuming the default, since composing {@code (1, ...)} for an unknown {@code k}
   * would be unsound.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The value of {@code k}, {@code 1} if {@code k} is omitted, or {@code null} if {@code k}
   *     is supplied but not a resolvable integer constant.
   */
  private Integer resolveK(PropagationCallGraphBuilder builder) {
    // Distinguish a genuinely omitted k (→ default 1) from one that is supplied but unresolvable
    // (→ ⊤). The synthetic method always has a k slot, so an empty points-to set alone can't tell
    // the two apart; check whether the call site actually passes k, positionally or by keyword
    // (the same idiom Input uses for its optional parameters).
    boolean kPassed =
        this.isKeywordArgumentPresent(builder, Parameters.K.getName())
            || this.getNumberOfPossiblePositionalArguments(builder).stream()
                .anyMatch(n -> n >= Parameters.K.getIndex() + 1);
    if (!kPassed) return 1; // k omitted; defaults to 1.
    OrdinalSet<InstanceKey> kPts =
        this.getArgumentPointsToSet(builder, Parameters.K.getIndex(), Parameters.K.getName());
    if (kPts == null || kPts.isEmpty()) return null; // k supplied but unresolvable → ⊤.
    for (Object value : getConstantValues(kPts, false))
      if (value instanceof Number) return ((Number) value).intValue();
    return null; // k supplied but not a constant int → ⊤.
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index == INDICES_INDEX) return EnumSet.of(DType.INT32);
    if (index != VALUES_INDEX)
      throw new IllegalArgumentException(
          "TopK has only 2 outputs (values, indices); got index " + index + ".");
    Set<DType> dtypes = null;
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    if (inputPts != null && !inputPts.isEmpty()) dtypes = this.getDTypesOfValue(builder, inputPts);
    if (dtypes == null || dtypes.isEmpty())
      // As with the shape, read the input's dtype in the caller's frame when its points-to set is
      // empty (an overlay-resolved input such as an elementwise binop). wala/ML#718.
      dtypes =
          this.getArgumentDTypesViaCallers(
              builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    // Fan out per (dtype, composed shape). The shape is input.shape[:-1] + (k,) (wala/ML#609); when
    // it can't be composed it's ⊤ (null dims).
    Set<DType> dtypes = this.getDTypesForIndex(builder, index);
    Set<List<Dimension<?>>> shapes = this.composedShapes(builder);
    Set<TensorType> ret = HashSetFactory.make();
    for (DType dt : dtypes) {
      String cellType = dt.name().toLowerCase(Locale.ROOT);
      if (shapes == null || shapes.isEmpty()) ret.add(new TensorType(cellType, null));
      else for (List<Dimension<?>> shape : shapes) ret.add(new TensorType(cellType, shape));
    }
    return ret;
  }

  /**
   * Aggregate {@code getTensorTypes} returns the UNION of per-index types, mirroring the
   * established {@link DatasetFromTensorsGenerator} convention. Concretely: {@code (values_type,
   * indices_type)} — values inherits input dtype with ⊤ shape; indices is {@code int32} with ⊤
   * shape.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return Union of per-index tensor types.
   */
  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<TensorType> ret = HashSetFactory.make();
    ret.addAll(this.getTensorTypesForIndex(builder, VALUES_INDEX));
    ret.addAll(this.getTensorTypesForIndex(builder, INDICES_INDEX));
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // A NamedTuple result has no single tensor shape, so the aggregate is ⊤; the per-element shape
    // (input.shape[:-1] + (k,)) is composed in getShapesForIndex (wala/ML#609). Composing here
    // instead would feed the wala/ML#480 attribute-access path, which reduces it to a wrong rank-0
    // shape.
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Aggregate dtype: union of values' (input dtype) and indices' (int32). The
    // TupleElementProvider
    // wrap on indexed access returns the precise per-index dtype; this aggregate is only consumed
    // when the caller doesn't index into the tuple.
    Set<DType> ret = EnumSet.noneOf(DType.class);
    ret.addAll(this.getDTypesForIndex(builder, VALUES_INDEX));
    ret.addAll(this.getDTypesForIndex(builder, INDICES_INDEX));
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
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
}
