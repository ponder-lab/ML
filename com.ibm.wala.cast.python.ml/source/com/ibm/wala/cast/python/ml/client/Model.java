package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
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

  /** Parameter positions and names for {@code tf.keras.Model}. */
  protected enum Parameters {
    /** The input(s) of the model. */
    INPUTS,
    /** The output(s) of the model. */
    OUTPUTS,
    /** Name of the model. */
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Model(PointsToSetVariable source) {
    super(source);
  }

  public Model(CGNode node) {
    super(node);
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
    String methodName = this.getNode().getMethod().getName().toString();
    if (methodName.equals("__call__") || methodName.equals("call") || methodName.equals("do")) {
      OrdinalSet<InstanceKey> inputPts =
          this.getArgumentPointsToSet(
              builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
      if (!inputPts.isEmpty()) return this.getShapesOfValue(builder, inputPts);
    }
    // TODO: Will need https://github.com/wala/ML/issues/340 to be resolved.
    return Collections.emptySet();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    String methodName = this.getNode().getMethod().getName().toString();
    if (methodName.equals("__call__") || methodName.equals("call") || methodName.equals("do")) {
      OrdinalSet<InstanceKey> inputPts =
          this.getArgumentPointsToSet(
              builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
      if (!inputPts.isEmpty()) return this.getDTypesOfValue(builder, inputPts);
    }
    // TODO: Will need https://github.com/wala/ML/issues/340 to be resolved.
    return EnumSet.noneOf(DType.class);
  }
}
