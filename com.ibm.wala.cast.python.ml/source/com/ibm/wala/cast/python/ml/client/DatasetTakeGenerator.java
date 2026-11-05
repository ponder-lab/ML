package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.take} (wala/ML#1216). The element is
 * the receiver's; the dataset has the least of {@code count} and the receiver's size, every element
 * when {@code count} is {@code -1}, and {@code count} of a receiver that repeats without end.
 */
public class DatasetTakeGenerator extends DatasetGenerator {

  /** The position of the {@code count} argument, excluding the receiver. */
  private static final int COUNT_PARAMETER_POSITION = 0;

  /** The keyword name of the {@code count} argument. */
  private static final String COUNT_PARAMETER_NAME = "count";

  public DatasetTakeGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetTakeGenerator(CGNode node) {
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
   * @implNote The least of each count the argument may be and each receiver size, the receiver's
   *     sizes for a count of {@code -1}, and the counts themselves where the receiver repeats
   *     without end. The sizes are unknown when the count does not resolve, when the receiver's do
   *     not, and when a {@code repeat} with a count is above the receiver, whose size a {@code
   *     repeat} does not count: it reads as its own receiver's.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    Set<Long> counts =
        getPossibleLongValues(
            this.getArgumentPointsToSet(builder, COUNT_PARAMETER_POSITION, COUNT_PARAMETER_NAME));
    if (counts == null || counts.isEmpty()) return Collections.emptySet();
    UpstreamRepeat repeat = this.upstreamRepeat(builder);
    if (repeat == UpstreamRepeat.COUNTED) return Collections.emptySet();
    boolean infinite = repeat == UpstreamRepeat.ENDLESS;
    Set<Long> sizes = infinite ? Collections.emptySet() : super.getDatasetSizes(builder);
    if (sizes.isEmpty() && !infinite) return Collections.emptySet();
    Set<Long> ret = HashSetFactory.make();
    for (Long count : counts) {
      // A count of -1 takes every element.
      if (count < 0) {
        if (infinite) return Collections.emptySet();
        ret.addAll(sizes);
      } else if (infinite) ret.add(count);
      else for (Long size : sizes) ret.add(Math.min(size, count));
    }
    return ret;
  }
}
