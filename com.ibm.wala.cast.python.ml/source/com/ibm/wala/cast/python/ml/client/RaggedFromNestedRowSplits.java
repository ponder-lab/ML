package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A representation of the `tf.RaggedTensor.from_nested_row_splits` API in TensorFlow. The
 * nested-ragged shape construction lives in {@link RaggedFromNested}; this form reads the {@code
 * nested_row_splits} argument and subtracts one from its first-element length ({@code row_splits}
 * has {@code nrows + 1} entries).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_row_splits">tf.RaggedTensor.from_nested_row_splits</a>.
 */
public class RaggedFromNestedRowSplits extends RaggedFromNested {

  public RaggedFromNestedRowSplits(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public RaggedFromNestedRowSplits(CGNode node) {
    super(node);
  }

  @Override
  protected String getNestedStructureParameterName() {
    return "nested_row_splits";
  }

  @Override
  protected Dimension<?> computeRowDim(Dimension<?> dim) {
    if (dim instanceof NumericDim) {
      return new NumericDim(((NumericDim) dim).value() - 1);
    }
    return null;
  }
}
