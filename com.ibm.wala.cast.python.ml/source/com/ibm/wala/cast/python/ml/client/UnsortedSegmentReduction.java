package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
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

  /** Positional index (excluding {@code self}) of the {@code segment_ids} argument. */
  private static final int SEGMENT_IDS_PARAMETER_POSITION = 1;

  /** Keyword name of the {@code segment_ids} argument. */
  private static final String SEGMENT_IDS_PARAMETER_NAME = "segment_ids";

  /** Positional index (excluding {@code self}) of the {@code num_segments} argument. */
  private static final int NUM_SEGMENTS_PARAMETER_POSITION = 2;

  /** Keyword name of the {@code num_segments} argument. */
  private static final String NUM_SEGMENTS_PARAMETER_NAME = "num_segments";

  public UnsortedSegmentReduction(PointsToSetVariable source) {
    super(source);
  }

  public UnsortedSegmentReduction(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "data";
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
            builder, NUM_SEGMENTS_PARAMETER_POSITION, NUM_SEGMENTS_PARAMETER_NAME);
    if (numSegments == null) return null;

    // The number of leading data axes consumed by segment_ids is segment_ids's rank.
    Integer segmentIdsRank =
        rankOf(
            this.shapesOfArg(builder, SEGMENT_IDS_PARAMETER_POSITION, SEGMENT_IDS_PARAMETER_NAME));
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
}
