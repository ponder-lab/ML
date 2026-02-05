package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by the <code>tf.keras.Model()</code> function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model">tf.keras.Model</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Model extends TensorGenerator {

  protected enum Parameters {
    INPUTS,
    OUTPUTS,
    NAME
  }

  public Model(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException(
        "Shape is mandatory and must be provided explicitly for tf.keras.Model.");
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException(
        "DType is mandatory and must be provided explicitly for tf.keras.Model.");
  }
}
