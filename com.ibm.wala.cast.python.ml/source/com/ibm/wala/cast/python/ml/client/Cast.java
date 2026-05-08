package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.cast(x, dtype, name=None)}. Intended output shape inherits from {@code
 * x}; output dtype is the explicit {@code dtype} argument (e.g., {@code tf.int32}, {@code
 * tf.float64}). The base-class dispatch in {@link TensorGenerator#getDTypes} reads the dtype
 * argument when {@link #getDTypeParameterPosition} returns a defined position, so this generator
 * just declares position 1 ({@code dtype}, after {@code x} at 0).
 *
 * <p><b>Current limitation</b>: in the as-shipped state, this {@code getDTypes} override is
 * bypassed because the {@code tf.cast} {@code pass_through} alias in {@code tensorflow.xml} wins
 * dispatch (the alias is intentionally retained &mdash; removing it breaks downstream tensor flow
 * for chained consumers like {@code reshape(cast(...))}; see <a
 * href="https://github.com/wala/ML/issues/509">wala/ML#509</a>). The analyzer therefore reports the
 * input dtype rather than the cast target, and {@code testCast} asserts the input dtype. Tracked by
 * <a href="https://github.com/wala/ML/issues/481">wala/ML#481</a>.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/cast">tf.cast</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Cast extends PassThroughUnaryTensorGenerator {

  public Cast(PointsToSetVariable source) {
    super(source);
  }

  public Cast(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0; // x — shape source
  }

  @Override
  protected String getInputParameterName() {
    return "x";
  }

  @Override
  protected int getDTypeParameterPosition() {
    return 1; // dtype — the explicit cast target
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }
}
