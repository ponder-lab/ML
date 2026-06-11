package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.linalg.trace}. Output dtype is inherited from the {@code x} input. Output
 * shape is the leading {@code x.shape[:-2]} (the trace collapses the last two dimensions to a
 * scalar per matrix in the batch). See wala/ML#449.
 *
 * <p>Extends {@link PassThroughUnaryTensorGenerator} for the dtype-from-input path; the shape
 * override below reuses the base's input-shape resolution and drops the last two dimensions.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/trace">tf.linalg.trace</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Trace extends PassThroughUnaryTensorGenerator {

  public Trace(PointsToSetVariable source) {
    super(source);
  }

  public Trace(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "x";
  }

  /**
   * Derives the output shape from the input {@code x}: {@code tf.linalg.trace} collapses the last
   * two dimensions (the trace of each matrix in the batch), so the result shape is {@code
   * x.shape[:-2]}. Reuses the superclass's input-shape resolution and drops the last two dimensions
   * from each candidate shape. A rank-2 input yields a scalar (rank-0) result.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when the input shape is unknown
   *     or its rank is below 2 for every candidate.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // The trace is defined over the last two axes, so the input must have rank >= 2.
      if (inputShape.size() < 2) continue;
      ret.add(new ArrayList<>(inputShape.subList(0, inputShape.size() - 2)));
    }
    return ret.isEmpty() ? null : ret;
  }
}
