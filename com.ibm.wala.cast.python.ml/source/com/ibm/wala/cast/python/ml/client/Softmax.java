package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * Generator for {@code tf.nn.softmax}. Produces a fresh tensor with the same shape and dtype as the
 * {@code logits} input (softmax is element-wise along one axis and shape-preserving).
 *
 * <p>Structurally identical to {@link Sigmoid}; differs only in the parameter name ({@code logits}
 * vs {@code x}). Extends the pass-through base so partial shape results ride through (wala/ML#718).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/softmax">tf.nn.softmax</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Softmax extends PassThroughUnaryTensorGenerator {

  /** Positional parameters of {@code tf.nn.softmax.do()}: {@code self logits axis name}. */
  private enum Parameters {
    /** The input tensor. */
    LOGITS;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs a {@code Softmax} from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     tf.nn.softmax(...)} invoke.
   */
  public Softmax(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code Softmax} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code softmax.do()} synthetic method.
   */
  public Softmax(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.LOGITS.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.LOGITS.getName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
