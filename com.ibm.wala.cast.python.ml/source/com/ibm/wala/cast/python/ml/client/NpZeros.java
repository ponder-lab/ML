package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for ndarrays created by NumPy's {@code zeros()} function.
 *
 * <p>Shares all shape and dtype logic with {@link NpOnes} (shape from the mandatory shape-tuple
 * argument, dtype from the {@code dtype} argument with a {@code float64} default), mirroring how
 * {@link Zeros} extends {@link Ones} on the TensorFlow side.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.zeros.html">numpy.zeros()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpZeros extends NpOnes {

  public NpZeros(PointsToSetVariable source) {
    super(source);
  }

  public NpZeros(CGNode node) {
    super(node);
  }
}
