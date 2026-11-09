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

  protected DatasetConcatenateGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetConcatenateGenerator(CGNode node) {
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
    List<TensorGenerator> arguments = this.getArgumentGenerators(builder);
    if (receivers == null || arguments == null) return null;
    List<TensorGenerator> ret = new ArrayList<>(receivers);
    for (TensorGenerator argument : arguments) if (!ret.contains(argument)) ret.add(argument);
    return ret;
  }

  /**
   * The datasets the concatenated dataset may be.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return Their generators, or {@code null} when the argument has no member or one does not
   *     resolve.
   */
  private List<TensorGenerator> getArgumentGenerators(PropagationCallGraphBuilder builder) {
    return this.generatorsOf(
        builder,
        this.getArgumentPointsToSet(builder, DATASET_PARAMETER_POSITION, DATASET_PARAMETER_NAME));
  }

  /**
   * The generators of the datasets a value may be.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param datasets The value's points-to set.
   * @return Their generators, without duplicates, or {@code null} when the value has no member or
   *     one does not resolve.
   */
  private List<TensorGenerator> generatorsOf(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> datasets) {
    if (datasets == null || datasets.isEmpty()) return null;
    List<TensorGenerator> ret = new ArrayList<>();
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
   * @implNote A concatenation has its receiver's elements and then its argument's: each size it may
   *     have is one size the receiver may have plus one the argument may have, each the union over
   *     the datasets it may be, so a dataset concatenated with itself counts twice and a receiver
   *     that may be either of two datasets counts one of them. The sizes are unknown when either's
   *     are.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    // A view fixed to one dataset is still the whole concatenation, so it has the same sizes.
    return sums(
        sizesOfAny(
            builder,
            this.generatorsOf(
                builder, this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF))),
        sizesOfAny(builder, this.getArgumentGenerators(builder)));
  }
}
