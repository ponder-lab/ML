package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.cast(x, dtype, name=None)}. Output shape inherits from {@code x}; output
 * dtype is the explicit {@code dtype} argument (e.g., {@code tf.int32}, {@code tf.float64}). The
 * base-class dispatch in {@link TensorGenerator#getDTypes} reads the dtype argument when {@link
 * #getDTypeParameterPosition} returns a defined position, so this generator just declares position
 * 1 ({@code dtype}, after {@code x} at 0).
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
