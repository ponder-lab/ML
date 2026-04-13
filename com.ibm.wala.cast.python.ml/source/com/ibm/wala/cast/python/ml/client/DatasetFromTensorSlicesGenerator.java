package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by {@code tf.data.Dataset.from_tensor_slices}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetFromTensorSlicesGenerator extends DatasetGenerator {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetFromTensorSlicesGenerator.class.getName());

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

  public DatasetFromTensorSlicesGenerator(CGNode node) {
    super(node);
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

    Set<List<Dimension<?>>> inputShapes = null;
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      inputShapes = this.getShapesOfValue(builder, tensorsPTS);
    }
    final int tensorsPTSSize = tensorsPTS == null ? -1 : tensorsPTS.size();
    final Set<List<Dimension<?>>> ptsPathShapes = inputShapes;
    LOGGER.fine(
        () ->
            "DatasetFromTensorSlicesGenerator.getDefaultShapes: source="
                + this.getSource()
                + ", tensorsPTS size="
                + tensorsPTSSize
                + ", inputShapes via pts-path="
                + (ptsPathShapes == null ? "null" : ptsPathShapes.size() + " shapes"));

    // Fallback: if the points-to set for the argument is empty (e.g., `tensors` is the result of
    // a Python binary op, for which WALA does not allocate a trackable target), walk the call
    // string to resolve the argument value number in each caller and delegate to getShapes, which
    // knows how to construct an ElementWiseOperation generator for binop-def'd locals.
    if (inputShapes == null || inputShapes.isEmpty()) {
      inputShapes =
          this.getArgumentShapesViaCallers(
              builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
      final Set<List<Dimension<?>>> fallbackShapes = inputShapes;
      LOGGER.fine(
          () ->
              "DatasetFromTensorSlicesGenerator.getDefaultShapes: fallback inputShapes="
                  + (fallbackShapes == null ? "null" : fallbackShapes.size() + " shapes"));
    }

    if (inputShapes == null) return null;
    if (inputShapes.isEmpty()) return Collections.emptySet();

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

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For from_tensor_slices, element dtypes are the same as the input tensor(s)' dtypes.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, tensorsPTS);
      if (!dtypes.isEmpty()) return dtypes;
    }

    // Fallback: walk the call string to resolve the argument in each caller.
    Set<DType> fallback =
        this.getArgumentDTypesViaCallers(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (fallback != null && !fallback.isEmpty()) return fallback;

    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, tensorsPTS);
      Set<Long> ret = HashSetFactory.make();
      for (List<Dimension<?>> shape : inputShapes) {
        if (!shape.isEmpty()) {
          Dimension<?> firstDim = shape.get(0);
          if (firstDim instanceof NumericDim) {
            ret.add(Long.valueOf(((NumericDim) firstDim).value()));
          }
        }
      }
      return ret;
    }
    return Collections.emptySet();
  }
}
