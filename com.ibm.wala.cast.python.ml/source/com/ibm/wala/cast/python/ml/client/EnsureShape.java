package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for the result of {@code tf.ensure_shape(x, shape)}: {@code x} itself, with the
 * declared static shape. The dtype is {@code x}'s. The shape is the declared one refined against
 * {@code x}'s: a known extent in {@code shape} gives the result's extent at that position, and a
 * {@code None} keeps {@code x}'s. When the declared shape does not resolve, the result has {@code
 * x}'s shape, which the run time only narrows.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/ensure_shape">tf.ensure_shape</a>
 */
public class EnsureShape extends TensorGenerator {

  /** The parameters of {@code tf.ensure_shape}, in positional order. */
  private enum Parameters {
    /** The tensor whose shape is ensured. */
    X,

    /** The declared shape, whose {@code None}s keep the input's extents. */
    SHAPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  public EnsureShape(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public EnsureShape(CGNode node) {
    super(node);
  }

  /**
   * The declared shapes, or {@code null} when the {@code shape} argument resolves none statically.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The declared shapes, or {@code null}.
   */
  private Set<List<Dimension<?>>> declaredShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());
    if (shapePts == null || !hasLiteralShapeEvidence(builder, shapePts)) return null;
    Set<List<Dimension<?>>> declared = this.getShapesFromShapeArgument(builder, shapePts);
    return declared == null || declared.isEmpty() ? null : declared;
  }

  /**
   * Refines an input shape by a declared one: the declared extent where it is known, the input's
   * where the declaration leaves it open. An input of another rank than the declaration's, which
   * the run time rejects, gives the declared shape.
   *
   * @param declared The declared shape.
   * @param input The input's shape, or {@code null} when it is unknown.
   * @return The refined shape.
   */
  private static List<Dimension<?>> refine(List<Dimension<?>> declared, List<Dimension<?>> input) {
    if (input == null || input.size() != declared.size()) return declared;
    List<Dimension<?>> ret = new ArrayList<>(declared.size());
    for (int i = 0; i < declared.size(); i++)
      ret.add(declared.get(i) instanceof NumericDim ? declared.get(i) : input.get(i));
    return ret;
  }

  /**
   * Whether a declared shape fixes every extent, so it is the result's whatever the input's is.
   *
   * @param declared The declared shape.
   * @return {@code true} iff every extent is a known number.
   */
  private static boolean isFullyDeclared(List<Dimension<?>> declared) {
    return declared.stream().allMatch(d -> d instanceof NumericDim);
  }

  /**
   * Refines each of the input's shapes by each declared shape.
   *
   * @param declared The declared shapes, or {@code null} when none resolves.
   * @param inputs The input's shapes, or {@code null} when they are unknown.
   * @return The result's shapes, or {@code null} when they are unknown: neither side is known, or
   *     the input's shape is unknown and the declaration leaves an extent open.
   */
  private static Set<List<Dimension<?>>> refineAll(
      Set<List<Dimension<?>>> declared, Set<List<Dimension<?>>> inputs) {
    if (declared == null) return inputs == null || inputs.isEmpty() ? null : inputs;
    if (inputs == null || inputs.isEmpty())
      // An extent the declaration leaves open is the input's, which is unknown here, so only a
      // fully
      // declared shape is the result's; otherwise the shape is unknown, and the feed refines the
      // input's dataflow shape instead.
      return declared.stream().allMatch(EnsureShape::isFullyDeclared) ? declared : null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> d : declared) for (List<Dimension<?>> i : inputs) ret.add(refine(d, i));
    return ret;
  }

  /**
   * The input's shapes: its points-to union first, then the per-context caller walk, as {@link
   * Reshape} reads its input.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The input's shapes, or {@code null} when they are unknown.
   */
  private Set<List<Dimension<?>>> inputShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    if (inputPts != null && !inputPts.isEmpty()) {
      Set<List<Dimension<?>>> fromValue = this.getShapesOfValue(builder, inputPts);
      if (fromValue != null && !fromValue.isEmpty()) return fromValue;
    }
    ShapeResult viaCallers =
        this.getArgumentShapeResultViaCallers(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    return viaCallers.members().isEmpty() ? null : viaCallers.members();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return refineAll(this.declaredShapes(builder), this.inputShapes(builder));
  }

  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapes(builder);
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return this.getDefaultDTypes(builder);
  }

  /**
   * The input's dtypes. The result is a tensor whatever the input reads, so an input whose dtype
   * nothing resolves gives the unknown dtype, as for {@link Reshape}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    if (inputPts != null && !inputPts.isEmpty()) {
      Set<DType> fromValue = this.getDTypesOfValue(builder, inputPts);
      if (fromValue != null && !fromValue.isEmpty()) return fromValue;
    }
    Set<DType> viaCallers =
        this.getArgumentDTypesViaCallers(
            builder, this.getValueParameterPosition(), this.getValueParameterName());
    if (viaCallers != null && !viaCallers.isEmpty()) return viaCallers;
    return EnumSet.of(DType.UNKNOWN);
  }

  /**
   * A transforming feed over the input, so an input typed only in the dataflow, such as a {@code
   * tf.function} parameter under an input signature, reaches the result refined by the declared
   * shape, as {@link Reshape}'s feed does.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    CGNode node = this.getNode();
    if (node == null || node.getIR() == null) return null;
    int inputVn =
        this.getArgumentValueNumber(
            builder, this.getValueParameterPosition(), this.getValueParameterName(), true);
    if (inputVn <= 0 || inputVn == Integer.MAX_VALUE) return null;
    PointerKey input =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, inputVn);
    Set<List<Dimension<?>>> declared = this.declaredShapes(builder);
    ShapeTransform rule = dims -> refineAll(declared, Collections.singleton(dims));
    return new TypeFeed(TypeFeedKind.TRANSFORM, List.of(input), rule);
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.ordinal();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  protected int getValueParameterPosition() {
    return Parameters.X.ordinal();
  }

  protected String getValueParameterName() {
    return Parameters.X.getName();
  }
}
