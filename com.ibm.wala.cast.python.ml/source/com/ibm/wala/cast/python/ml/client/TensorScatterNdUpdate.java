package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.tensor_scatter_nd_update}. The op returns a copy of {@code tensor} with
 * specific entries updated, so output shape and dtype are both equal to {@code tensor}'s — a true
 * shape-and-dtype passthrough on the first input. Inherits the entire passthrough behavior from
 * {@link PassThroughUnaryTensorGenerator}; only the input-arg identification needs to be supplied.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update">tf.tensor_scatter_nd_update</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorScatterNdUpdate extends PassThroughUnaryTensorGenerator {

  public TensorScatterNdUpdate(PointsToSetVariable source) {
    super(source);
  }

  public TensorScatterNdUpdate(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "tensor";
  }
}
