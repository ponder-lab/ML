package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for the graph message-passing aggregation ops {@code tf.math.unsorted_segment_sum},
 * {@code tf.math.unsorted_segment_max}, and {@code tf.math.unsorted_segment_mean}. These reduce a
 * {@code data} tensor of shape {@code [N, ...]} into {@code [num_segments, ...]} by aggregating the
 * rows that share a {@code segment_ids} value.
 *
 * <p>Output dtype inherits from the {@code data} (arg 0) input. Output shape is left at ⊤: the
 * leading axis becomes the runtime {@code num_segments} value (typically a {@code tf.shape(...)}
 * result rather than a static constant), so forwarding {@code data}'s shape would be unsound on the
 * first axis. Dtype is the load-bearing axis here; shape can follow. See <a
 * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum">tf.math.unsorted_segment_sum</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class UnsortedSegmentReduction extends PassThroughUnaryTensorGenerator {

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
   * Returns ⊤ (unknown shape). The output's leading axis is the runtime {@code num_segments} value,
   * which is not statically known, so the shape is left unknown rather than mis-forwarding {@code
   * data}'s shape. Dtype still inherits from {@code data} via the inherited {@link
   * #getDefaultDTypes}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null} (⊤).
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
