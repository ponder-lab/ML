package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.concatenate} (wala/ML#1215). The
 * element is either its receiver's element or its argument's, so both are datasets the element
 * follows, and a read of the element joins over them.
 */
public class DatasetConcatenateGenerator extends DatasetJoinGenerator {

  /** The position of the concatenated dataset in a {@code concatenate} call. */
  private static final int DATASET_PARAMETER_POSITION = 0;

  /** The name of the concatenated dataset's parameter. */
  private static final String DATASET_PARAMETER_NAME = "dataset";

  public DatasetConcatenateGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetConcatenateGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The receiver's datasets and the concatenated dataset's, since the element is either
   *     one's element.
   */
  @Override
  public List<TensorGenerator> getReceiverGenerators(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> receivers = super.getReceiverGenerators(builder);
    if (receivers == null) return null;
    OrdinalSet<InstanceKey> datasets =
        this.getArgumentPointsToSet(builder, DATASET_PARAMETER_POSITION, DATASET_PARAMETER_NAME);
    if (datasets == null || datasets.isEmpty()) return null;
    List<TensorGenerator> ret = new ArrayList<>(receivers);
    for (InstanceKey dataset : datasets) {
      TensorGenerator generator = receiverGenerator(builder, dataset);
      if (generator == null) return null;
      if (!ret.contains(generator)) ret.add(generator);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A concatenation has as many elements as its datasets together: each size it may have
   *     is the sum of one size of each, and unknown when any dataset's is.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDatasetSizes(builder);
    return sumOfSizes(builder, datasets);
  }
}
