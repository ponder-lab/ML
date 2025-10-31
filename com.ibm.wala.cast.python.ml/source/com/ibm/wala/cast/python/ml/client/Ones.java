package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by the `ones()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/ones">TensorFlow ones() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Ones extends TensorGenerator {

  private static final java.util.logging.Logger LOGGER = getLogger(Ones.class.getName());

  private static final String FUNCTION_NAME = "tf.ones()";

  private static final int SHAPE_PARAMETER_POSITION = 0;

  private static final int DTYPE_PARAMETER_POSITION = 1;

  public Ones(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    LOGGER.info(
        "No dtype specified for source: " + source + ". Using default dtype of: " + FLOAT32 + " .");

    // Use the default dtype of float32.
    return EnumSet.of(FLOAT32);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException("Shape is mandatory and must be provided explicitly.");
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }
}
