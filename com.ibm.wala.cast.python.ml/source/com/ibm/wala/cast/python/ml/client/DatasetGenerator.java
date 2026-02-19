package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE;
import static com.ibm.wala.cast.python.util.Util.getFunction;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
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
    TypeReference func = getFunction(this.getSource());
    if (func != null && func.equals(DATASET_FROM_TENSOR_SLICES_TYPE)) {
      // For tf.data.Dataset.from_tensor_slices(tensors), the dataset elements are created by
      // slicing the input tensors along their first dimension. Thus, the element shapes are
      // the input shapes with the first dimension removed.
      // The 'tensors' argument is at position 0 (args: this, tensors, name).
      OrdinalSet<InstanceKey> tensorsPTS = this.getArgumentPointsToSet(builder, 0, "tensors");
      if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
        Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, tensorsPTS);
        Set<List<Dimension<?>>> ret = HashSetFactory.make();
        for (List<Dimension<?>> shape : inputShapes) {
          if (shape.size() > 0) {
            // Remove the first dimension to account for slicing.
            ret.add(new ArrayList<>(shape.subList(1, shape.size())));
          } else {
            // If the input is already a scalar (unexpected for from_tensor_slices),
            // the element shape is empty.
            ret.add(Collections.emptyList());
          }
        }
        return ret;
      }
    }

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
    TypeReference func = getFunction(this.getSource());
    if (func != null && func.equals(DATASET_FROM_TENSOR_SLICES_TYPE)) {
      // For from_tensor_slices, element dtypes are the same as the input tensor(s)' dtypes.
      // The 'tensors' argument is at position 0 (args: this, tensors, name).
      OrdinalSet<InstanceKey> tensorsPTS = this.getArgumentPointsToSet(builder, 0, "tensors");
      if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
        Set<DType> dTypes = this.getDTypesOfValue(builder, tensorsPTS);
        if (!dTypes.isEmpty()) {
          return EnumSet.copyOf(dTypes);
        }
      }
    }

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
