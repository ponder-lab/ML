package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.argmin}. Inherits {@link Argmax}'s ⊤ shape (deferred to wala/ML#462)
 * and fixed {@link com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType#INT64} output dtype —
 * the shape and dtype semantics are identical to {@code tf.math.argmax}; only the runtime value
 * differs (smallest vs. largest index). See wala/ML#449 (Tier 6).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/argmin">tf.math.argmin</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Argmin extends Argmax {

  public Argmin(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Argmin(CGNode node) {
    super(node);
  }
}
