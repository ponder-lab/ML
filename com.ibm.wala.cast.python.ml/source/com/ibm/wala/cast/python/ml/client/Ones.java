package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for tensors created by the `ones()` function in TensorFlow. The explicit-shape +
 * optional-dtype allocator machinery lives in {@link ShapeAndDTypeAllocator}; {@code tf.ones} adds
 * no value semantics relevant to tensor-type inference.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/ones">TensorFlow ones() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Ones extends ShapeAndDTypeAllocator {

  public Ones(PointsToSetVariable source) {
    super(source);
  }

  public Ones(CGNode node) {
    super(node);
  }
}
