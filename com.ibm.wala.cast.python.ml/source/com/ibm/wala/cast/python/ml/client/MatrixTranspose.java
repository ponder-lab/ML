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
 * Generator for {@code tf.linalg.matrix_transpose}. Output dtype is inherited from the {@code a}
 * input. Output shape swaps the last two dimensions (the matrix transpose acts on the final two
 * axes, preserving any leading batch dimensions), so a {@code (..., M, N)} input yields {@code
 * (..., N, M)}. Previously modeled as a first-argument {@code pass_through}, which reported the
 * input shape unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket
 * 2a.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/linalg/matrix_transpose">tf.linalg.matrix_transpose</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class MatrixTranspose extends PassThroughUnaryTensorGenerator {

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public MatrixTranspose(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public MatrixTranspose(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "a";
  }

  /**
   * Derives the output shape from the input by swapping its last two dimensions; leading batch
   * dimensions are preserved. A {@code (..., M, N)} input yields {@code (..., N, M)}. Reuses the
   * superclass's input-shape resolution.
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
      // The matrix transpose is defined over the last two axes, so the input must have rank >= 2.
      if (inputShape.size() < 2) continue;
      List<Dimension<?>> transposed = new ArrayList<>(inputShape);
      int last = transposed.size() - 1;
      Dimension<?> tmp = transposed.get(last);
      transposed.set(last, transposed.get(last - 1));
      transposed.set(last - 1, tmp);
      ret.add(transposed);
    }
    return ret.isEmpty() ? null : ret;
  }
}
