package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/** A generator for tensors created by {@code tf.data.Dataset.from_tensors}. */
public class DatasetFromTensorsGenerator extends DatasetGenerator {

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

  public DatasetFromTensorsGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetFromTensorsGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For tf.data.Dataset.from_tensors(tensors), the dataset contains a single element
    // which is the 'tensors' argument itself.
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    LOGGER.info(
        "DatasetFromTensorsGenerator.getDefaultShapes: tensorsPTS="
            + (tensorsPTS != null ? tensorsPTS.size() : "null"));
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = this.getShapesOfValue(builder, tensorsPTS);
      LOGGER.info("DatasetFromTensorsGenerator.getDefaultShapes: ret=" + ret);
      return ret;
    }
    return Collections.emptySet();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    LOGGER.info(
        "DatasetFromTensorsGenerator.getDefaultDTypes: tensorsPTS="
            + (tensorsPTS != null ? tensorsPTS.size() : "null"));
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> ret = this.getDTypesOfValue(builder, tensorsPTS);
      LOGGER.info("DatasetFromTensorsGenerator.getDefaultDTypes: ret=" + ret);
      return ret;
    }
    return Collections.emptySet();
  }

  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(1L);
  }
}
