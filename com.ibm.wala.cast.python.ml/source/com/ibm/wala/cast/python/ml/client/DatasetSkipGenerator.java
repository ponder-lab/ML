package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.skip} (wala/ML#1214). The element is
 * the receiver's; the dataset has {@code count} fewer elements, none when {@code count} is {@code
 * -1} or at least the receiver's size.
 */
public class DatasetSkipGenerator extends DatasetGenerator {

  /** The position of the {@code count} argument, excluding the receiver. */
  private static final int COUNT_PARAMETER_POSITION = 0;

  /** The keyword name of the {@code count} argument. */
  private static final String COUNT_PARAMETER_NAME = "count";

  public DatasetSkipGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetSkipGenerator(CGNode node) {
    super(node);
  }

  /** The element is the receiver's, so a receiver that may be several datasets is spread. */
  @Override
  public boolean elementFollowsReceiver() {
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Each receiver size less each count the argument may be, at least zero. The sizes are
   *     unknown when the receiver's are or the count does not resolve.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    Set<Long> sizes = super.getDatasetSizes(builder);
    if (sizes.isEmpty()) return sizes;
    Set<Long> counts =
        getPossibleLongValues(
            this.getArgumentPointsToSet(builder, COUNT_PARAMETER_POSITION, COUNT_PARAMETER_NAME));
    if (counts == null || counts.isEmpty()) return Collections.emptySet();
    Set<Long> ret = HashSetFactory.make();
    for (Long size : sizes)
      for (Long count : counts)
        // A count of -1 skips the whole dataset.
        ret.add(count < 0 ? 0L : Math.max(0L, size - count));
    return ret;
  }
}
