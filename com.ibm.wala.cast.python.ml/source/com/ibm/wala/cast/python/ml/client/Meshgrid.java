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
 * Generator for {@code tf.meshgrid(*xi, indexing='xy')}. Returns an N-tuple of tensors (where
 * {@code N == len(xi)}); all output tensors share the broadcast of all input shapes and the first
 * input's dtype.
 *
 * <p>Implements {@link TupleElementProvider} so that destructuring (e.g. {@code X, Y =
 * tf.meshgrid(x, y)}) resolves to the right per-element shape and dtype regardless of how many
 * indices the user accesses. The XML allocates only a 1-element list (post-wala/ML#380 inlining);
 * the {@link TupleElementProvider} wrap in {@link TensorGeneratorFactory#getGenerator} answers any
 * index with the per-element type, even when the XML-side tuple is undersized.
 *
 * <p>Output shape is currently ⊤ — computing the precise broadcast shape requires reading every
 * input's shape (PA-resolvable for many cases; bounded by {@code len(xi)}). A follow-up can compose
 * the per-input shapes. For now the precise per-element dtype (first input's) is the load-bearing
 * fix vs. the previous {@code ReadDataFallback} routing.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/meshgrid">tf.meshgrid</a>
 * @see <a href="https://github.com/wala/ML/issues/449">wala/ML#449</a> (Tier 5).
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Meshgrid extends TensorGenerator implements TupleElementProvider {

  /**
   * Parameter positions and keyword names for {@code tf.meshgrid(*xi, indexing='xy')}. Meshgrid
   * uses pure varargs &mdash; the first user-facing positional argument is the first tensor in the
   * {@code *args} varpack &mdash; so {@code Parameters.ARGS.getIndex() == 0} resolves to that first
   * tensor. {@code INDEXING} captures the optional keyword argument that selects between Cartesian
   * (`'xy'`) and matrix (`'ij'`) ordering; the analyzer does not consume it.
   */
  protected enum Parameters {
    /** First input tensor in the {@code *args} varargs; the dtype source. */
    ARGS,

    /** Keyword controlling Cartesian (`'xy'`) vs matrix (`'ij'`) ordering; not consumed. */
    INDEXING;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
     *
     * @return The lowercased enum name (e.g. {@code "args"}).
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

  public Meshgrid(PointsToSetVariable source) {
    super(source);
  }

  public Meshgrid(CGNode node) {
    super(node);
  }

  /** Always yields a tuple — meshgrid's signature guarantees a tuple-of-tensors return. */
  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    // All outputs share the broadcast of input shapes; emit ⊤ until the broadcast composer
    // lands.
    return null;
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(builder, Parameters.ARGS.getIndex(), Parameters.ARGS.getName());
    if (inputPts == null || inputPts.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = this.getDTypesOfValue(builder, inputPts);
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    // Shapes are uniformly ⊤ until the broadcast composer lands; emit one TensorType per
    // dtype with null dims. When shapes become precise, this method needs to fan out per shape.
    Set<DType> dtypes = this.getDTypesForIndex(builder, index);
    Set<TensorType> ret = HashSetFactory.make();
    for (DType dt : dtypes) ret.add(new TensorType(dt.name().toLowerCase(Locale.ROOT), null));
    return ret;
  }

  /**
   * Aggregate {@code getTensorTypes} returns the per-element type (all elements share the same type
   * for meshgrid). Matches the {@link DatasetFromTensorsGenerator} convention of unioning per-index
   * types — for meshgrid the "union" collapses to a single type since every element has the same
   * shape and dtype.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The element tensor type (all N outputs share this).
   */
  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    return this.getTensorTypesForIndex(builder, 0);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null; // ⊤ at every level until broadcast composer lands.
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.getDTypesForIndex(builder, 0);
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
