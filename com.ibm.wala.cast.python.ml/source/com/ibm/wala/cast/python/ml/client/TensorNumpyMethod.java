package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for a tensor's {@code numpy()} method, {@code t.numpy()} (wala/ML#1105). The result is
 * a NumPy array of the tensor's shape and dtype, so both pass through from the receiver, and its
 * origin is {@link TensorOrigin#NUMPY}.
 *
 * <p>Tensor allocations carry no method summaries, so the call has no call-graph target; the
 * factory recognizes it by its function value, a {@code numpy} attribute read off a tensor.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/Tensor#numpy">tf.Tensor.numpy</a>
 */
public class TensorNumpyMethod extends PassThroughUnaryTensorGenerator {

  public TensorNumpyMethod(PointsToSetVariable source) {
    super(source);
  }

  public TensorNumpyMethod(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return RECEIVER_PARAMETER_POSITION;
  }

  @Override
  protected String getInputParameterName() {
    return null;
  }

  /**
   * Returns the producing library of the modeled value: a NumPy array (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
