package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;

/**
 * A generator for weights created by {@code tf.keras.layers.Layer.add_weight()}, the universal
 * Keras weight-creation API (typically called from the lazily-invoked {@code build()}, <a
 * href="https://github.com/wala/ML/issues/595">wala/ML#595</a>). The name-first signature {@code
 * (name, shape, dtype, ...)} puts the explicit shape at position 1 and the dtype at position 2,
 * unlike the shape-first allocator default. The {@code float32} default dtype inherited from {@link
 * TensorTypeAllocator} matches the Keras default variable dtype. See <a
 * href="https://github.com/wala/ML/issues/667">wala/ML#667</a>.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Layer#add_weight">TensorFlow
 *     add_weight() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class AddWeight extends TensorTypeAllocator {

  /**
   * Constructs a new {@link AddWeight} generator from a points-to set source.
   *
   * @param source The points-to set variable representing the source of the weight tensor.
   */
  public AddWeight(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a new "manual" {@link AddWeight} generator from a call graph node.
   *
   * @param node The call graph node representing the {@code add_weight()} call.
   */
  public AddWeight(CGNode node) {
    super(node);
  }

  /**
   * Returns the position of the {@code shape} parameter in the name-first {@code add_weight(name,
   * shape, dtype, ...)} signature.
   *
   * @return 1, the position of the {@code shape} parameter.
   */
  @Override
  protected int getShapeParameterPosition() {
    return 1;
  }

  /**
   * Returns the position of the {@code dtype} parameter in the name-first {@code add_weight(name,
   * shape, dtype, ...)} signature.
   *
   * @return 2, the position of the {@code dtype} parameter.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return 2;
  }

  /**
   * {@inheritDoc}
   *
   * <p>The positions this generator declares ({@link #getShapeParameterPosition()}, {@link
   * #getDTypeParameterPosition()}) are counted in the USER's frame, {@code add_weight(name, shape,
   * dtype, ...)} with {@code name} at 0, because a source anchor reads the user's own call. The
   * Keras layer-call generators count in the trampoline's frame instead ({@code func self inputs}:
   * the instance at 0, the first argument at 1), which is why the shared resolution reads a
   * trampoline caller's invoke as it is. A manual anchor on the {@code add_weight} summary node
   * reads at that trampoline, whose invoke carries the bound instance before the user's positional
   * arguments, so the user's position {@code p} is the trampoline's {@code p + 1}; the shift is
   * this generator's alone, not the shared path's (wala/ML#996). A keyword resolves by name on
   * either path and is unaffected.
   */
  @Override
  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    boolean manual = this.getInvokeInstruction() == null && this.getSource() == null;
    return super.getArgumentPointsToSet(
        builder, manual && paramPos >= 0 ? paramPos + 1 : paramPos, paramName);
  }
}
