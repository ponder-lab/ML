package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
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
 * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>. It degrades to ⊤ when {@code k} is
 * not a resolvable constant or the input shape is unknown rank or rank-0. The per-element dtype
 * (FLOAT32 for values, INT32 for indices) is resolved independently.
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
   * shape and the {@code k} argument (default {@code 1}) and replaces the last axis with {@code k}.
   * Returns ⊤ ({@code null}) if {@code k} is not a resolvable constant or any input shape is
   * unknown rank or rank-0. See <a href="https://github.com/wala/ML/issues/609">wala/ML#609</a>.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of composed output shapes, or {@code null} (⊤) if it can't be composed.
   */
  private Set<List<Dimension<?>>> composedShapes(PropagationCallGraphBuilder builder) {
    Integer k = this.resolveK(builder);
    if (k == null) return null;
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    if (inputPts == null || inputPts.isEmpty()) return null;
    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);
    if (inputShapes == null || inputShapes.isEmpty()) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> in : inputShapes) {
      // Unknown rank (null) or a rank-0 scalar can't have its last axis replaced; degrade to ⊤.
      if (in == null || in.isEmpty()) return null;
      List<Dimension<?>> out = new ArrayList<>(in);
      out.set(out.size() - 1, new NumericDim(k));
      ret.add(out);
    }
    // inputShapes is non-empty and the loop returns ⊤ for any null/empty shape, so ret is
    // populated.
    return ret;
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
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    if (inputPts == null || inputPts.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = this.getDTypesOfValue(builder, inputPts);
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
