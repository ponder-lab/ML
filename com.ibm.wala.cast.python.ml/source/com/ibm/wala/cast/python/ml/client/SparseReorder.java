package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A representation of the `tf.sparse.reorder()` API in TensorFlow. The result is a {@code
 * SparseTensor} with the shape and dtype of its {@code sp_input} (<a
 * href="https://github.com/wala/ML/issues/1351">wala/ML#1351</a>).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/sparse/reorder">tf.sparse.reorder
 *     API</a>.
 */
public class SparseReorder extends ConvertToTensor {

  @Override
  protected boolean producesSparseTensor() {
    return true;
  }

  protected enum Parameters {
    SP_INPUT,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  protected SparseReorder(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  protected SparseReorder(CGNode node) {
    super(node);
  }

  @Override
  protected int getValueParameterPosition() {
    return Parameters.SP_INPUT.getIndex();
  }

  @Override
  protected String getValueParameterName() {
    return Parameters.SP_INPUT.getName();
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
