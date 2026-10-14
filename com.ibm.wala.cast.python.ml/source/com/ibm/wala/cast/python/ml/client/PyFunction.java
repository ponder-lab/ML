package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.py_function(func, inp, Tout, name=None)}: as for {@link NumpyFunction},
 * the result's dtype is the one {@code Tout}, at the same position, states, and its shape is
 * unknown.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/py_function">
 *     tf.py_function</a>
 */
public class PyFunction extends NumpyFunction {

  public PyFunction(PointsToSetVariable source) {
    super(source);
  }

  public PyFunction(CGNode node) {
    super(node);
  }
}
