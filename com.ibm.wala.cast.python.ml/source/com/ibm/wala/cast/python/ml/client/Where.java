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
 * per-element from {@code x} or {@code y} based on {@code condition}. Output shape is the broadcast
 * of all three; output dtype matches {@code x} (and {@code y}, which TF requires to be the same
 * dtype as {@code x}).
 *
 * <p>The 1-argument form {@code tf.where(condition)} is semantically different &mdash; it returns
 * {@code int64} indices of the true entries &mdash; and has no test fixture today. This generator
 * targets the common 3-argument form. For the 1-argument form, the union over {@code x} and {@code
 * y} below resolves to ⊤ (both args are absent), which is a sound fallback.
 *
 * <p>Shape and dtype are produced by unioning the inferred sets over {@code x} <em>and</em> {@code
 * y}. TF requires {@code x.dtype == y.dtype} and broadcasts {@code condition}, {@code x}, {@code y}
 * to a common shape, so at runtime the union is a singleton; under static imprecision (e.g., when
 * one of {@code x} / {@code y} is the result of a binary op whose PTS is empty) the two
 * argument-side sets can disagree, and unioning ensures we don't drop {@code y}'s contribution.
 * {@code condition}'s shape is intentionally not unioned in: in idiomatic user code its shape
 * equals {@code x} / {@code y}'s, but it can also be a strictly-broader-rank bool mask and
 * including it would over-approximate the common case.
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
    if ((xShapes == null || xShapes.isEmpty()) && (yShapes == null || yShapes.isEmpty()))
      return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    if (xShapes != null) ret.addAll(xShapes);
    if (yShapes != null) ret.addAll(yShapes);
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> xDTypes = dtypesOfArg(builder, Parameters.X);
    Set<DType> yDTypes = dtypesOfArg(builder, Parameters.Y);
    if ((xDTypes == null || xDTypes.isEmpty()) && (yDTypes == null || yDTypes.isEmpty()))
      return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    if (xDTypes != null) ret.addAll(xDTypes);
    if (yDTypes != null) ret.addAll(yDTypes);
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
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
