package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.linalg.adjoint}. The conjugate transpose swaps the last two dimensions
 * exactly like {@link MatrixTranspose} (it differs only in conjugating the entries, which doesn't
 * affect shape or — for the real dtypes modeled here — dtype), so the shape logic is inherited. The
 * input argument is named {@code matrix} rather than {@code a}. Previously modeled as a
 * first-argument {@code pass_through}. See <a
 * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/adjoint">tf.linalg.adjoint</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Adjoint extends MatrixTranspose {

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Adjoint(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Adjoint(CGNode node) {
    super(node);
  }

  @Override
  protected String getInputParameterName() {
    return "matrix";
  }
}
