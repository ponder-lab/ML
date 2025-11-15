package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Represents a call to the <code>constant()</code> function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/constant">constant()</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Constant extends TensorGenerator {

  private static final String FUNCTION_NAME = "tf.constant()";

  private static final int VALUE_PARAMETER_POSITION = 0;

  private static final int DTYPE_PARAMETER_POSITION = 1;

  private static final int SHAPE_PARAMETER_POSITION = 2;

  public Constant(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If the shape argument is not specified, then the shape is inferred from the shape of value.
    // TODO: Handle keyword arguments.
    return getShapes(builder, this.getValueArgumentValueNumber());
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If the dtype argument is not specified, then the type is inferred from the type of value.
    // TODO: Handle keyword arguments.
    return getDTypes(builder, this.getValueArgumentValueNumber());
  }

  protected int getValueArgumentValueNumber() {
    return getArgumentValueNumber(this.getValueParameterPosition());
  }

  protected int getValueParameterPosition() {
    return VALUE_PARAMETER_POSITION;
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
