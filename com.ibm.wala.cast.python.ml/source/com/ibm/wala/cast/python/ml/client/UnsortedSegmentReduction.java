package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
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
 * shape is recovered when {@code num_segments} is a static constant and the {@code data}/{@code
 * segment_ids} shapes (hence {@code segment_ids}'s rank) are known; otherwise — notably when {@code
 * num_segments} is a runtime value (e.g. a {@code tf.shape(...)} result, the common case in a GNN
 * where it is the node count) — the shape is left at ⊤ rather than mis-forwarding {@code data}'s
 * shape. See <a href="https://github.com/wala/ML/issues/570">wala/ML#570</a> and <a
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
   * Computes the output shape {@code [num_segments] ++ data.shape[segment_ids.ndim:]} when {@code
   * num_segments} resolves to a static constant and the {@code data}/{@code segment_ids} shapes are
   * known, and returns ⊤ ({@code null}) otherwise. The leading axis becomes the constant {@code
   * num_segments}; the trailing axes are {@code data}'s dimensions after the ones consumed by
   * {@code segment_ids} (whose rank is read from its shape). Dtype still inherits from {@code data}
   * via the inherited {@link #getDefaultDTypes}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of output shapes, or {@code null} (⊤) when {@code num_segments} is not a static
   *     constant or the input shapes / {@code segment_ids}'s rank cannot be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The leading output axis is num_segments; recoverable only when it is a static constant. A
    // runtime num_segments (e.g. a node count via tf.shape) leaves the shape at ⊤.
    Integer numSegments =
        this.resolveStaticIntArgument(
            builder, Parameters.NUM_SEGMENTS.getIndex(), Parameters.NUM_SEGMENTS.getName());
    if (numSegments == null) return null;

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
      outputShape.add(new NumericDim(numSegments));
      outputShape.addAll(dataShape.subList(segmentIdsRank, dataShape.size()));
      result.add(outputShape);
    }
    return result;
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
}
