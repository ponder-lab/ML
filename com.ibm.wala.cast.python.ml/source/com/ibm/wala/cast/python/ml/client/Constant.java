package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Represents a call to the <code>constant()</code> function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/constant">constant()</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Constant extends TensorGenerator {

  protected enum Parameters {
    VALUE,
    DTYPE,
    SHAPE,
    NAME,
    VERIFY_SHAPE;

    public String getParameterName() {
      return name().toLowerCase();
    }

    public int getParameterIndex() {
      return ordinal();
    }
  }

  public Constant(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If the shape argument is not specified, then the shape is inferred from the shape of value.
    // TODO: Handle keyword arguments.
    return this.getShapes(builder, this.getValueArgumentValueNumber());
  }

  /**
   * {@inheritDoc}
   *
   * <p>If the <code>dtype</code> argument is not specified, then the type is inferred from the type
   * of value.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // TODO: Handle keyword arguments.
    return getDTypes(builder, this.getValueArgumentValueNumber());
  }

  protected int getValueArgumentValueNumber() {
    return getArgumentValueNumber(this.getValueParameterPosition());
  }

  protected int getValueParameterPosition() {
    return Parameters.VALUE.getParameterIndex();
  }

  protected String getValueParameterName() {
    return Parameters.VALUE.getParameterName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getParameterIndex();
  }

  protected String getShapeParameterName() {
    return Parameters.SHAPE.getParameterName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getParameterIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getParameterName();
  }
}
