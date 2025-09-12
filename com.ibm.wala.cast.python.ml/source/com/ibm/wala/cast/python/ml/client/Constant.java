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

  private static final int VALUE_NUMBER_FOR_VALUE_ARGUMENT = 2;

  private static final int VALUE_NUMBER_FOR_DTYPE_ARGUMENT = 3;

  private static final int VALUE_NUMBER_FOR_SHAPE_ARGUMENT = 4;

  public Constant(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If the shape argument is not specified, then the shape is inferred from the shape of value.
    // TODO: Handle keyword arguments.
    return getShapes(builder, this.getValueNumberForValueArgument());
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If the dtype argument is not specified, then the type is inferred from the type of value.
    // TODO: Handle keyword arguments.
    return getDTypes(builder, this.getValueNumberForValueArgument());
  }

  @Override
  protected int getValueNumberForDTypeArgument() {
    return VALUE_NUMBER_FOR_DTYPE_ARGUMENT;
  }

  protected int getValueNumberForValueArgument() {
    return VALUE_NUMBER_FOR_VALUE_ARGUMENT;
  }

  @Override
  protected int getValueNumberForShapeArgument() {
    // Shapes can also be specified as an explicit argument. Here, we examine the third explicit
    // argument (recall that the first argument is implicit and corresponds to the called
    // function's name).
    return VALUE_NUMBER_FOR_SHAPE_ARGUMENT;
  }
}
