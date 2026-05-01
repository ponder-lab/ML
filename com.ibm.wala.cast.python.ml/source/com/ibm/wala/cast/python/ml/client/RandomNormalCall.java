package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for the {@code __call__} on a {@code tf.initializers.RandomNormal} instance. A call
 * {@code instance(shape, dtype=None)} returns a tensor of the requested shape drawn from the normal
 * distribution.
 *
 * <p>Differs from {@link Normal} only in the positional offset of {@code shape} and {@code dtype}
 * within the invoke: a {@code __call__} invoke passes the instance as positional arg 0 (self), so
 * {@code shape} is at positional arg 1 and {@code dtype} at positional arg 2, whereas {@code
 * tf.random.normal(shape, ...)} has {@code shape} at positional arg 0. Inheriting from {@link
 * Normal} reuses the shape/dtype resolution logic unchanged; only the arg positions change.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomNormal">tf.keras.initializers.RandomNormal</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RandomNormalCall extends Normal {

  /**
   * Constructs a {@code RandomNormalCall} from a caller-side {@link PointsToSetVariable} (the
   * result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code tf.initializers.RandomNormal} instance.
   */
  public RandomNormalCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code RandomNormalCall} anchored to a manual node. Used when the factory is
   * invoked without a caller-side {@link PointsToSetVariable} (e.g., from {@code
   * createManualGenerator}).
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public RandomNormalCall(CGNode node) {
    super(node);
  }

  /**
   * @return The positional index of the {@code shape} argument in the {@code __call__} invoke (one
   *     past {@code self}, which occupies position 0).
   */
  @Override
  protected int getShapeParameterPosition() {
    return 1;
  }

  /**
   * @return The parameter name {@code "shape"} &mdash; matches the XML signature in {@code
   *     tensorflow.xml}.
   */
  @Override
  protected String getShapeParameterName() {
    return "shape";
  }

  /**
   * @return The positional index of the {@code dtype} argument in the {@code __call__} invoke (two
   *     past {@code self}; one past {@code shape}).
   */
  @Override
  protected int getDTypeParameterPosition() {
    return 2;
  }

  /**
   * @return The parameter name {@code "dtype"} &mdash; matches the XML signature in {@code
   *     tensorflow.xml}.
   */
  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }
}
