package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * A generator for {@code np.expand_dims(a, axis)} (<a
 * href="https://github.com/wala/ML/issues/1308">wala/ML#1308</a>): the input's shape with a size-1
 * axis inserted at the constant {@code axis}, and the input's dtype, as {@link ExpandDims} computes
 * them for {@code tf.expand_dims}. The two functions differ in the input's keyword, which is {@code
 * a} here, and in what they return, which is an ndarray here. An {@code axis} that names several
 * positions, which NumPy accepts as a tuple, gives an unknown shape.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html">
 *     numpy.expand_dims</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpExpandDims extends ExpandDims {

  protected NpExpandDims(PointsToSetVariable source) {
    super(source);
  }

  protected NpExpandDims(CGNode node) {
    super(node);
  }

  @Override
  protected String getInputParameterName() {
    return "a";
  }

  /**
   * The input's dtype, which an array or a tensor keeps. A Python number, list or tuple is
   * converted as NumPy converts it, which the input's read does not give, so the dtype is unknown
   * (wala/ML#1309).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The result's dtypes.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // TODO: Read such an input as `np.array` reads it (wala/ML#1309).
    if (this.inputMayBePythonNumberOrSequence(builder)) return EnumSet.of(DType.UNKNOWN);
    return super.getDefaultDTypes(builder);
  }

  /**
   * The feed of the TensorFlow counterpart, and none over an input that may be a Python number,
   * list or tuple, whose dtype the conversion decides (wala/ML#1309).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The feed over the caller-side input keys, or {@code null} for such an input or when no
   *     input key is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    if (this.inputMayBePythonNumberOrSequence(builder)) return null;
    return super.getTypeFeed(builder);
  }

  /**
   * Returns the producing library of the modeled value: an {@code np.expand_dims(...)} call, so the
   * value is an ndarray whatever its input is (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
