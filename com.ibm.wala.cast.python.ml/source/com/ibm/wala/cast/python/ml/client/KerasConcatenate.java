package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for the functional {@code tf.keras.layers.concatenate(inputs, axis=-1)} spelling. The
 * computation is {@link Concat}'s: every entry of the tensors list contributes its extent along the
 * resolved axis and the rest of the shape carries over from the first entry. Only the argument
 * naming and the default axis differ: the list parameter is named {@code inputs}, and an unsupplied
 * axis is {@code -1} where {@code tf.concat}'s is {@code 0} — wiring the Keras spelling to the
 * {@code tf.concat} summary would therefore concatenate default calls along the wrong axis (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>).
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/concatenate">tf.keras.layers.concatenate</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class KerasConcatenate extends Concat {

  /** The Keras concatenate family's default axis, the last one. */
  protected static final int KERAS_DEFAULT_AXIS = -1;

  /** The tensors-list argument's keyword name in the Keras spellings. */
  protected static final String INPUTS_PARAMETER_NAME = "inputs";

  /**
   * Constructs a {@code KerasConcatenate} from a caller-side {@link PointsToSetVariable} (the
   * result of the call).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     tf.keras.layers.concatenate} invoke.
   */
  public KerasConcatenate(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code KerasConcatenate} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public KerasConcatenate(CGNode node) {
    super(node);
  }

  /**
   * @return {@value #INPUTS_PARAMETER_NAME}; the Keras spellings name the tensors list {@code
   *     inputs}.
   */
  @Override
  protected String getValuesParameterName() {
    return INPUTS_PARAMETER_NAME;
  }

  /**
   * @return {@value #KERAS_DEFAULT_AXIS}; the Keras concatenate family defaults to the last axis.
   */
  @Override
  protected int getDefaultAxis() {
    return KERAS_DEFAULT_AXIS;
  }
}
