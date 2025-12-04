package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.util.Util.getFunction;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import java.util.List;
import java.util.Set;

/**
 * A representation of the `tf.ragged.constant()` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/ragged/constant">tf.ragged.constant</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RaggedConstant extends ZerosLike {

  protected enum Parameters {
    PYLIST,
    DTYPE,
    RAGGED_RANK,
    INNER_SHAPE,
    NAME,
    ROW_SPLITS_DTYPE
  }

  public RaggedConstant(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected String getSignature() {
    TypeReference function = getFunction(this.getSource());
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // TODO Auto-generated method stub
    return super.getDefaultShapes(builder);
  }
}
