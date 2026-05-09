package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
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
 * <p>Output shape precision is currently ⊤ — computing the precise {@code input.shape[:-1] + (k,)}
 * requires reading the input's shape (PA-resolvable for many cases) and the {@code k} argument
 * (PA-constant in the common case). A follow-up can wire that in. For now the per-element dtype
 * precision (FLOAT32 for values, INT32 for indices) is the load-bearing fix vs. the previous {@code
 * ReadDataFallback} routing, which produced {@code ⊤}/{@code unknown} for both.
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
    // Both values and indices share shape input.shape[:-1] + (k,). Currently emit ⊤ until an
    // input-shape + k composer is added.
    return null;
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
    // Shapes are uniformly ⊤ until the input-shape + k composer lands; emit one TensorType per
    // dtype with null dims. When shapes become precise, this method needs to fan out per shape.
    Set<DType> dtypes = this.getDTypesForIndex(builder, index);
    Set<TensorType> ret = HashSetFactory.make();
    for (DType dt : dtypes) ret.add(new TensorType(dt.name().toLowerCase(), null));
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
    // ⊤ at the aggregate level — per-index access returns the precise shape (also currently ⊤
    // pending the input-shape + k composer; both axes line up for now).
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
