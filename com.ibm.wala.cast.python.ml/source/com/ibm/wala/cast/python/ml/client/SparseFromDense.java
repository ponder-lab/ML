package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A representation of the `tf.sparse.from_dense()` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/sparse/from_dense">tf.sparse.from_dense
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SparseFromDense extends ConvertToTensor {

  @Override
  protected boolean producesSparseTensor() {
    return true;
  }

  protected enum Parameters {
    TENSOR,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseFromDense(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public SparseFromDense(CGNode node) {
    super(node);
  }

  @Override
  protected int getValueParameterPosition() {
    return Parameters.TENSOR.getIndex();
  }

  @Override
  protected String getValueParameterName() {
    return Parameters.TENSOR.getName();
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
