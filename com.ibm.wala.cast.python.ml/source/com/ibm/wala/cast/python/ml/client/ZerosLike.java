package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by the `zeros_like()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/zeros_like">TensorFlow zeros_like()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ZerosLike extends Constant {

  public static final String FUNCTION_NAME = "tf.zeros_like()";

  /**
   * The shape argument is not explicitly provided to zeros_like(); rather, the shape is inferred
   * from the `input` argument.
   */
  private static final int SHAPE_PARAMETER_POSITION = -1;

  public ZerosLike(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapesFromShapeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    throw new UnsupportedOperationException(
        "Shapes are derived from the `input` argument and cannot be provided explicitly.");
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
