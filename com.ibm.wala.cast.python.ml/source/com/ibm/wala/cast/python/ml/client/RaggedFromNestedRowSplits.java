package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowSplits.Parameters.NESTED_ROW_SPLITS;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.logging.Logger;

/**
 * A representation of the `tf.RaggedTensor.from_nested_row_splits` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_row_splits">tf.RaggedTensor.from_nested_row_splits</a>.
 */
public class RaggedFromNestedRowSplits extends RaggedFromNestedRowLengths {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromNestedRowSplits.class.getName());

  protected enum Parameters {
    FLAT_VALUES,
    NESTED_ROW_SPLITS,
    NAME,
    VALIDATE
  }

  public RaggedFromNestedRowSplits(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected OrdinalSet<InstanceKey> getNestedStructurePointsToSet(
      PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder,
        this.getNestedRowLengthsParameterPosition(),
        NESTED_ROW_SPLITS.name().toLowerCase());
  }

  @Override
  protected Dimension<?> computeRowDim(Dimension<?> dim) {
    if (dim instanceof NumericDim) {
      return new NumericDim(((NumericDim) dim).value() - 1);
    }
    return null;
  }
}
