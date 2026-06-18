package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A representation of the `tf.RaggedTensor.from_nested_row_lengths` API in TensorFlow. The
 * nested-ragged shape construction lives in {@link RaggedFromNested}; this form reads the {@code
 * nested_row_lengths} argument and uses its first-element length directly as the row count.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_row_lengths">tf.RaggedTensor.from_nested_row_lengths</a>.
 */
public class RaggedFromNestedRowLengths extends RaggedFromNested {

  public RaggedFromNestedRowLengths(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected String getNestedStructureParameterName() {
    return "nested_row_lengths";
  }

  @Override
  protected Dimension<?> computeRowDim(Dimension<?> dim) {
    return dim;
  }
}
