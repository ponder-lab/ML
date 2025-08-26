package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptyList;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
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

  private static final int VALUE_NUMBER_FOR_SHAPE_ARGUMENT = 4;

  private static final int VALUE_NUMBER_FOR_DTYPE_ARGUMENT = 3;

  private static final int VALUE_NUMBER_FOR_VALUE_ARGUMENT = 2;

  public Constant(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // The shape is that of the first explicit argument.
    // TODO: Handle keyword arguments.
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, 2);

    for (InstanceKey valueIK : pointerAnalysis.getPointsToSet(valuePK))
      if (valueIK instanceof ConstantKey)
        // It's a scalar value. A scalar has no dimensions, so its shape is represented by an
        // empty tuple ().
        ret.add(emptyList());
      else
        // TODO: More cases.
        throw new IllegalStateException(
            "Expected a " + ConstantKey.class + " for value, but got: " + valueIK + ".");

    return ret;
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If the dtype argument is not specified, then the type is inferred from the type of value.
    // TODO: Handle keyword arguments.
    return getDTypes(builder, this.getValueNumberForValueArgument());
  }

  @Override
  protected int getValueNumberForShapeArgument() {
    // Shapes can also be specified as an explicit argument. Here, we examine the third explicit
    // argument (recall that the first argument is implicit and corresponds to the called
    // function's name).
    return VALUE_NUMBER_FOR_SHAPE_ARGUMENT;
  }

  @Override
  protected int getValueNumberForDTypeArgument() {
    return VALUE_NUMBER_FOR_DTYPE_ARGUMENT;
  }

  protected int getValueNumberForValueArgument() {
    return VALUE_NUMBER_FOR_VALUE_ARGUMENT;
  }
}
