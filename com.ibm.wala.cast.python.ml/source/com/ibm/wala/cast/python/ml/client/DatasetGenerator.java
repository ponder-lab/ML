package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by <code>tf.data.Dataset</code> transformations.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetGenerator extends TensorGenerator {

  public DatasetGenerator(PointsToSetVariable source) {
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
    // For dataset transformations, default to shapes of the input dataset (the receiver).
    // The receiver is 'self' (arg0 in IR).
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, "self");
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      return this.getShapesOfValue(builder, receiverPTS);
    }
    throw new UnsupportedOperationException(
        "Modeling for tf.data.Dataset transformation " + this.getSource() + " is missing.");
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For dataset transformations, default to dtypes of the input dataset (the receiver).
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, "self");
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      Set<DType> dTypes = this.getDTypesOfValue(builder, receiverPTS);
      if (!dTypes.isEmpty()) {
        return EnumSet.copyOf(dTypes);
      }
    }
    throw new UnsupportedOperationException(
        "Modeling for tf.data.Dataset transformation " + this.getSource() + " is missing.");
  }
}
