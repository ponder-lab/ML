package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.from_tensor_slices}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetFromTensorSlicesGenerator extends DatasetGenerator {

  protected enum Parameters {
    TENSORS,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetFromTensorSlicesGenerator(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For tf.data.Dataset.from_tensor_slices(tensors), the dataset elements are created by
    // slicing the input tensors along their first dimension. Thus, the element shapes are
    // the input shapes with the first dimension removed.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
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
    // If we can't find the argument, we can't infer shape.
    return Collections.emptySet();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For from_tensor_slices, element dtypes are the same as the input tensor(s)' dtypes.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      return this.getDTypesOfValue(builder, tensorsPTS);
    }
    return Collections.emptySet();
  }
}
