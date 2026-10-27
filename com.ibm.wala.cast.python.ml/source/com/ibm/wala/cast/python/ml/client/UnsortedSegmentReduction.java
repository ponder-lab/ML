package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for the graph message-passing aggregation ops {@code tf.math.unsorted_segment_sum},
 * {@code tf.math.unsorted_segment_max}, and {@code tf.math.unsorted_segment_mean}. These reduce a
 * {@code data} tensor of shape {@code [N, ...]} into {@code [num_segments, ...]} by aggregating the
 * rows that share a {@code segment_ids} value.
 *
 * <p>Output dtype inherits from the {@code data} (arg 0) input. Output shape is {@code
 * [num_segments] ++ data.shape[segment_ids.ndim:]}: the leading axis is {@code num_segments} and
 * the trailing axes are {@code data}'s axes after the ones consumed by {@code segment_ids}. That
 * shape is recovered when the {@code data}/{@code segment_ids} shapes (hence {@code segment_ids}'s
 * rank) are known, and is left at ⊤ otherwise rather than mis-forwarding {@code data}'s shape. A
 * {@code num_segments} that is not a static constant, notably a runtime value such as a {@code
 * tf.shape(...)} result, the common case in a GNN where it is the node count, still gives the
 * leading axis, of a size the analysis did not compute (wala/ML#1160). See <a
 * href="https://github.com/wala/ML/issues/570">wala/ML#570</a> and <a
 * href="https://github.com/wala/ML/issues/582">wala/ML#582</a>.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum">tf.math.unsorted_segment_sum</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class UnsortedSegmentReduction extends PassThroughUnaryTensorGenerator {

  /**
   * Positional parameters (after {@code self}) of {@code tf.math.unsorted_segment_*}: {@code data
   * segment_ids num_segments name}. Only the three that participate in shape inference are modeled;
   * the trailing {@code name} is omitted.
   */
  private enum Parameters {
    /** The values tensor being aggregated; supplies the output's trailing axes and its dtype. */
    DATA,

    /**
     * The per-row segment assignments; its rank is the number of leading {@code data} axes the
     * reduction consumes.
     */
    SEGMENT_IDS,

    /** The number of output segments; its static value becomes the output's leading axis. */
    NUM_SEGMENTS;

    /**
     * @return The lowercase keyword-parameter name (e.g., {@code "segment_ids"}), suitable for
     *     {@link TensorGenerator#getArgumentPointsToSet}.
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    /**
     * @return The zero-based positional index (after {@code self}), suitable for {@link
     *     TensorGenerator#getArgumentPointsToSet}.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public UnsortedSegmentReduction(PointsToSetVariable source) {
    super(source);
  }

  public UnsortedSegmentReduction(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.DATA.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.DATA.getName();
  }

  /**
   * Computes the output shape {@code [num_segments] ++ data.shape[segment_ids.ndim:]} when the
   * {@code data}/{@code segment_ids} shapes are known, and returns ⊤ ({@code null}) otherwise. The
   * leading axis is {@code num_segments}'s static value, or, when it has none, the axis {@link
   * #unresolvedLeadingAxis} gives (wala/ML#1160); the trailing axes are {@code data}'s dimensions
   * after the ones consumed by {@code segment_ids} (whose rank is read from its shape). Dtype still
   * inherits from {@code data} via the inherited {@link #getDefaultDTypes}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of output shapes, or {@code null} (⊤) when the input shapes or {@code
   *     segment_ids}'s rank cannot be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The leading output axis is num_segments. A num_segments that is not a static constant, such
    // as a node count read with tf.shape, still adds that one axis, so the rank and the trailing
    // axes are kept (wala/ML#1160).
    Integer numSegments =
        this.resolveStaticIntArgument(
            builder, Parameters.NUM_SEGMENTS.getIndex(), Parameters.NUM_SEGMENTS.getName());
    Dimension<?> leading =
        numSegments != null ? new NumericDim(numSegments) : this.unresolvedLeadingAxis(builder);

    // The number of leading data axes consumed by segment_ids is segment_ids's rank.
    Integer segmentIdsRank =
        rankOf(
            this.shapesOfArg(
                builder, Parameters.SEGMENT_IDS.getIndex(), Parameters.SEGMENT_IDS.getName()));
    if (segmentIdsRank == null) return null;

    Set<List<Dimension<?>>> dataShapes =
        this.shapesOfArg(builder, this.getInputParameterPosition(), this.getInputParameterName());
    if (dataShapes == null || dataShapes.isEmpty()) return null;

    Set<List<Dimension<?>>> result = HashSetFactory.make();
    for (List<Dimension<?>> dataShape : dataShapes) {
      // segment_ids indexes data's leading axes, so data must be at least that rank.
      if (dataShape == null || dataShape.size() < segmentIdsRank) return null;
      List<Dimension<?>> outputShape = new ArrayList<>(dataShape.size() - segmentIdsRank + 1);
      outputShape.add(leading);
      outputShape.addAll(dataShape.subList(segmentIdsRank, dataShape.size()));
      result.add(outputShape);
    }
    return result;
  }

  /**
   * The leading output axis when {@code num_segments} is not a static constant (wala/ML#1160), by
   * the criterion of wala/ML#721, which follows TensorFlow's static shape.
   *
   * <ul>
   *   <li>An element of a tensor's shape, such as {@code tf.shape(x)[0]}, is the dimension it
   *       reads: a number when {@code x}'s axis is one, as TensorFlow folds it at trace time; a
   *       {@link DynamicDim} when that axis is dynamic; and an {@link UnresolvedDim} otherwise. It
   *       is resolved where the call is made, through the parameters that forward it.
   *   <li>Failing that, a tensor is a {@link DynamicDim}, since TensorFlow's static shape reports
   *       {@code None} for a tensor whose value is not known at trace time.
   *   <li>Any other value, such as a Python {@code int} the analysis did not compute, is a fixed
   *       size, an {@link UnresolvedDim}.
   * </ul>
   *
   * @param builder The propagation call graph builder.
   * @return The leading axis.
   */
  private Dimension<?> unresolvedLeadingAxis(PropagationCallGraphBuilder builder) {
    int index = Parameters.NUM_SEGMENTS.getIndex();
    String name = Parameters.NUM_SEGMENTS.getName();
    if (this.getSource() != null) {
      int vn = this.getArgumentValueNumber(builder, index, name, true);
      // `Integer.MAX_VALUE` is the synthetic generator function's sentinel, not a value number.
      if (vn > 0 && vn != Integer.MAX_VALUE) {
        Dimension<?> element =
            this.resolveElementDimension(builder, this.getNode(), vn, name, MAX_FORWARDING_DEPTH);
        if (element instanceof NumericDim || element instanceof DynamicDim) return element;
        if (element != null) return UnresolvedDim.INSTANCE;
      }
    }
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, index, name);
    if (pts != null)
      for (InstanceKey ik : pts)
        if (TensorFlowTypes.TENSOR_TYPE.equals(ik.concreteType().getReference()))
          return DynamicDim.INSTANCE;
    return UnresolvedDim.INSTANCE;
  }

  /** The most parameters a shape element is followed through to the call that computes it. */
  private static final int MAX_FORWARDING_DEPTH = 4;

  /**
   * The dimension a value reads as an element of a tensor's shape, such as {@code tf.shape(x)[0]}
   * (wala/ML#1160), following a parameter to the arguments its callers pass, as the summary's
   * {@code num_segments} parameter and a helper's own parameter forward it.
   *
   * @param builder The propagation call graph builder.
   * @param node The node whose IR defines {@code vn}.
   * @param vn The value's value number in {@code node}'s IR.
   * @param keyword The keyword a caller may pass the parameter by, or {@code null}.
   * @param depth How many more parameters to follow.
   * @return The element's dimension, or {@code null} when some caller's value isn't an element of a
   *     resolvable shape.
   */
  private Dimension<?> resolveElementDimension(
      PropagationCallGraphBuilder builder, CGNode node, int vn, String keyword, int depth) {
    if (node.getIR() == null || node.getDU() == null) return null;
    Dimension<?> element =
        this.resolveShapeVectorElementDim(builder, node, node.getIR().getSymbolTable(), vn);
    if (element != null || depth == 0) return element;

    // A parameter: the dimension is the join of the callers' arguments.
    int parameter = -1;
    for (int i = 0; i < node.getIR().getNumberOfParameters(); i++)
      if (node.getIR().getParameter(i) == vn) parameter = i;
    if (parameter < 0) return null;

    Set<Dimension<?>> dimensions = HashSetFactory.make();
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, node)) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction call)) return null;
      // A synthetic caller, such as a method trampoline, need not pass its arguments in the slots
      // of the body's parameters.
      if (callerInvoke.fst.getMethod().isWalaSynthetic()) return null;
      int argument = -1;
      if (keyword != null) argument = call.getUse(keyword);
      if (argument == -1) {
        // A starred argument at or before the parameter's slot unpacks a sequence of unknown
        // length, so the slot's use is not the argument the parameter binds (wala/ML#751).
        for (int starred : call.getStarredPositions()) if (starred <= parameter) return null;
        if (parameter < call.getNumberOfPositionalParameters()) argument = call.getUse(parameter);
      }
      if (argument <= 0) return null;
      Dimension<?> dimension =
          this.resolveElementDimension(builder, callerInvoke.fst, argument, null, depth - 1);
      if (dimension == null) return null;
      dimensions.add(dimension);
    }
    if (dimensions.isEmpty()) return null;
    if (dimensions.size() == 1) return dimensions.iterator().next();
    return dimensions.contains(DynamicDim.INSTANCE) ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE;
  }

  /**
   * Returns the common rank (number of dimensions) shared by every candidate shape, or {@code null}
   * when the rank cannot be pinned down — an empty/⊤ shape set, a candidate of unknown rank, or
   * candidates that disagree on rank.
   *
   * @param shapes The candidate shapes whose rank to resolve.
   * @return The shared rank, or {@code null} if it is not unambiguously known.
   */
  private static Integer rankOf(Set<List<Dimension<?>>> shapes) {
    if (shapes == null || shapes.isEmpty()) return null;
    Integer rank = null;
    for (List<Dimension<?>> shape : shapes) {
      if (shape == null) return null;
      if (rank != null && rank != shape.size()) return null;
      rank = shape.size();
    }
    return rank;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform until a member-wise upgrade.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * This generator transforms its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }
}
