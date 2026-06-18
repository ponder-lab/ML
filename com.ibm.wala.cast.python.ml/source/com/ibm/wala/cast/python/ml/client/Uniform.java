package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A representation of the `random.uniform()` function in TensorFlow. The shape-first random
 * signature (shape at position 0, dtype at position 3) and float32 default come from {@link
 * RandomDistribution}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/random/uniform">TensorFlow uniform()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Uniform extends RandomDistribution {

  public Uniform(PointsToSetVariable source) {
    super(source);
  }

  public Uniform(CGNode node) {
    super(node);
  }
}
