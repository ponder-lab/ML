package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for the 3-argument form of {@code tf.where(condition, x, y, name=None)}, which selects
 * per-element from {@code x} or {@code y} based on {@code condition}. Output shape is the broadcast
 * of all three; output dtype matches {@code x} (and {@code y}, which TF requires to be the same
 * dtype as {@code x}).
 *
 * <p>The 1-argument form {@code tf.where(condition)} is semantically different — it returns {@code
 * int64} indices of the true entries — and has no test fixture today. This generator targets the
 * common 3-argument form. For the 1-argument form, the inherited shape passthrough on {@code x}
 * will return ⊤ (since {@code x} is absent), which is a sound fallback.
 *
 * <p>This is the dtype-only-passthrough variant: shape inheritance from {@code x} is sound only
 * when {@code condition} doesn't broadcast to a different shape than {@code x}. The common case in
 * user code has {@code condition.shape == x.shape == y.shape} (e.g., {@code tf.where(x &gt; 0, x,
 * -x)}), so the {@code x}-shape passthrough is sound for that case.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/where">tf.where</a>
 * @see <a href="https://github.com/wala/ML/issues/422">wala/ML#422</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Where extends PassThroughUnaryTensorGenerator {

  /**
   * Argument-position constant for the dtype/shape source. Position 0 is {@code condition} (a bool
   * tensor); position 1 is {@code x}, the dtype source for the result.
   */
  private static final int X_POSITION = 1;

  /** Keyword name for {@code x}. */
  private static final String X_NAME = "x";

  public Where(PointsToSetVariable source) {
    super(source);
  }

  public Where(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return X_POSITION;
  }

  @Override
  protected String getInputParameterName() {
    return X_NAME;
  }
}
