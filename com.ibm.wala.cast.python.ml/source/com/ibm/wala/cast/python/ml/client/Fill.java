package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * A representation of the TensorFlow <code>fill()</code> function.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/fill">TensorFlow fill() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Fill extends Constant {

  private static final int VALUE_NUMBER_FOR_SHAPE_ARGUMENT = 2;

  private static final int VALUE_NUMBER_FOR_VALUE_ARGUMENT = 3;

  /**
   * The dtype argument is not explicitly provided to fill(); rather, the dtype is inferred from the
   * `value` argument.
   */
  private static final int VALUE_NUMBER_FOR_DTYPE_ARGUMENT = -1;

  public Fill(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected int getValueNumberForDTypeArgument() {
    return VALUE_NUMBER_FOR_DTYPE_ARGUMENT;
  }

  @Override
  protected int getValueNumberForValueArgument() {
    return VALUE_NUMBER_FOR_VALUE_ARGUMENT;
  }

  @Override
  protected int getValueNumberForShapeArgument() {
    return VALUE_NUMBER_FOR_SHAPE_ARGUMENT;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException("Shape is mandatory and must be provided explicitly.");
  }
}
