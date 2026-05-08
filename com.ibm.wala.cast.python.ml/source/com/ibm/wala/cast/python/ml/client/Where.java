package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for the 3-argument form of {@code tf.where(condition, x, y, name=None)}, which selects
 * per-element from {@code x} or {@code y} based on {@code condition}.
 *
 * <p><b>TensorFlow's runtime contract</b> is that the output shape is the broadcast of all three
 * inputs and the output dtype matches {@code x} (with TF requiring {@code y} to share {@code x}'s
 * dtype). This generator does <em>not</em> compute a broadcast shape: it produces shape and dtype
 * by unioning the inferred sets over {@code x} <em>and</em> {@code y} only, and intentionally
 * ignores {@code condition}'s shape. Under TF's runtime contract the three sets agree, so the union
 * collapses to a singleton; the modeling sacrifices a strictly-broader-rank bool-mask {@code
 * condition} (which would broadcast the output beyond {@code x} / {@code y}'s shape) for a closer
 * fit on the common case where {@code condition.shape == x.shape == y.shape}.
 *
 * <p>The reason for the union over {@code x} / {@code y} (rather than reading either alone) is
 * static imprecision: under the analyzer's lattice, the inferred sets for {@code x} and {@code y}
 * can disagree (e.g., when one is the result of a binary op whose PTS is empty). Unioning ensures
 * we don't drop {@code y}'s contribution.
 *
 * <p>The 1-argument form {@code tf.where(condition)} is semantically different &mdash; it returns
 * {@code int64} indices of the true entries &mdash; and has no test fixture today. This generator
 * targets the common 3-argument form. For the 1-argument form, the union over {@code x} and {@code
 * y} below resolves to ⊤ (both args are absent), which is a sound fallback.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/where">tf.where</a>
 * @see <a href="https://github.com/wala/ML/issues/422">wala/ML#422</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Where extends TensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.where(condition, x, y, name=None)}.
   * Ordinals match the position in the XML's {@code paramNames} after the implicit {@code self}
   * receiver, so {@code Parameters.CONDITION.getIndex() == 0} resolves to the first user-facing
   * positional argument.
   */
  protected enum Parameters {
    /** Boolean mask selecting between {@code x} (true) and {@code y} (false), per-element. */
    CONDITION,

    /** Tensor whose value is taken where {@code condition} is true; dtype/shape source. */
    X,

    /** Tensor whose value is taken where {@code condition} is false; dtype/shape source. */
    Y,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in {@link #getArgumentPointsToSet} / similar arg-resolution
     * helpers when the call site uses {@code keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "condition"}).
     */
    public String getName() {
      return name().toLowerCase();
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

  public Where(PointsToSetVariable source) {
    super(source);
  }

  public Where(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> xShapes = shapesOfArg(builder, Parameters.X);
    Set<List<Dimension<?>>> yShapes = shapesOfArg(builder, Parameters.Y);
    // Lattice: `null` means ⊤ (unknown). If either side is ⊤, the union is ⊤ &mdash; taking only
    // the other side's concrete shapes would be an unsound narrowing (we'd claim the result is
    // exactly that shape when the unknown side could be any shape).
    if (xShapes == null || yShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.addAll(xShapes);
    ret.addAll(yShapes);
    // Empty propagates as-is: empty set = ⊥ (provably not a tensor). Don't collapse to `null`
    // (⊤) &mdash; that would conflate ⊥ with ⊤.
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> xDTypes = dtypesOfArg(builder, Parameters.X);
    Set<DType> yDTypes = dtypesOfArg(builder, Parameters.Y);
    // Same lattice rule as `getDefaultShapes`: `null` means ⊤. If either side is ⊤, the result
    // is ⊤ &mdash; collapse to `{UNKNOWN}` rather than narrow to the concrete side.
    if (xDTypes == null || yDTypes == null) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    ret.addAll(xDTypes);
    ret.addAll(yDTypes);
    // Lattice-normalize: if the union contains UNKNOWN, collapse to exactly `{UNKNOWN}` (⊤).
    // Mixed sets like `{FLOAT32, UNKNOWN}` would violate the dtype lattice convention.
    if (ret.contains(DType.UNKNOWN)) return EnumSet.of(DType.UNKNOWN);
    // Empty propagates as-is: empty set = ⊥ (provably not a tensor). Don't collapse to
    // `{UNKNOWN}` (⊤) &mdash; that would conflate ⊥ with ⊤.
    return ret;
  }

  /**
   * PTS-first arg-shape resolver with caller-walk fallback. Mirrors {@link
   * PassThroughUnaryTensorGenerator}'s private helper but takes a {@link Parameters} enum.
   *
   * @param builder The propagation call graph builder.
   * @param param The argument to resolve.
   * @return The resolved shapes, or {@code null} if neither path recovers.
   */
  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, Parameters param) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(builder, param.getIndex(), param.getName());
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, param.getIndex(), param.getName());
  }

  /**
   * Dtype counterpart of {@link #shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param param The argument to resolve.
   * @return The resolved dtypes, or {@code null} if neither path recovers.
   */
  private Set<DType> dtypesOfArg(PropagationCallGraphBuilder builder, Parameters param) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(builder, param.getIndex(), param.getName());
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    return this.getArgumentDTypesViaCallers(builder, param.getIndex(), param.getName());
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
