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
 * Generator for {@code tf.linalg.diag}. Output dtype is inherited from the {@code diagonal} input.
 * Output shape places the input's last axis on the diagonal of a new trailing square, so a {@code
 * (..., M)} input yields {@code (..., M, M)} (rank increases by one). Previously modeled as a
 * first-argument {@code pass_through}, which reported the input shape unchanged. See <a
 * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/diag">tf.linalg.diag</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Diag extends PassThroughUnaryTensorGenerator {

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Diag(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Diag(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "diagonal";
  }

  /**
   * Derives the output shape from the input by appending a copy of its last dimension, forming a
   * trailing square whose diagonal is the input's last axis. A {@code (..., M)} input yields {@code
   * (..., M, M)}. Reuses the superclass's input-shape resolution.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when the input shape is unknown
   *     or its rank is below 1 for every candidate.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      // The diagonal must have at least one axis to place on a square.
      if (inputShape.isEmpty()) continue;
      List<Dimension<?>> withDiag = new ArrayList<>(inputShape);
      withDiag.add(inputShape.get(inputShape.size() - 1));
      ret.add(withDiag);
    }
    return ret.isEmpty() ? null : ret;
  }
}
