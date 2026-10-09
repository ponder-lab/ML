package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code tf.abs(x, name=None)}. The shape is {@code x}'s. The dtype is {@code x}'s
 * for a real input, while a complex input's magnitude is real: {@code complex64} gives {@code
 * float32} and {@code complex128} gives {@code float64}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/abs">tf.math.abs</a>
 */
public class Abs extends PassThroughUnaryTensorGenerator {

  public Abs(PointsToSetVariable source) {
    super(source);
  }

  public Abs(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "x";
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (DType dtype : super.getDefaultDTypes(builder)) ret.add(magnitude(dtype));
    return ret;
  }

  /**
   * The dtype of the magnitude of a value of the given dtype.
   *
   * @param dtype The input's dtype.
   * @return The real dtype of a complex one's width, else the dtype itself.
   */
  private static DType magnitude(DType dtype) {
    if (dtype == DType.COMPLEX64) return DType.FLOAT32;
    if (dtype == DType.COMPLEX128) return DType.FLOAT64;
    return dtype;
  }

  /**
   * The input's shape passes through, and so does its dtype, unless the input reads as complex: the
   * magnitude then has another dtype, so the feed carries the shape alone and the result's own
   * dtype stands. An input whose dtype is known only from dataflow state keeps the pass-through
   * feed, which is the reading for every real input.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The feed over the caller-side input keys, or {@code null} when none is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    Set<DType> input = super.getDefaultDTypes(builder);
    boolean complex = input.contains(DType.COMPLEX64) || input.contains(DType.COMPLEX128);
    return this.getTypeFeed(builder, complex ? TypeFeedKind.SHAPE_ONLY : TypeFeedKind.PASS_THROUGH);
  }
}
