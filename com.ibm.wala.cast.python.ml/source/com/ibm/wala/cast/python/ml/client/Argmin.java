package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.math.argmin}. Reuses {@link Argmax}'s axis-removal shape inference and
 * fixed {@link com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType#INT64} output dtype (the
 * shape and dtype semantics are identical between the two ops; only the runtime value differs). See
 * wala/ML#449 (Tier 6).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/argmin">tf.math.argmin</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Argmin extends Argmax {

  public Argmin(PointsToSetVariable source) {
    super(source);
  }
}
