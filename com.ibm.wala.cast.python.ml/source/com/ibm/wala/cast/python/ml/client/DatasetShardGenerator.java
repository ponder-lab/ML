package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.shard} (wala/ML#1214). The element is
 * the receiver's; the dataset holds every {@code num_shards}th element starting at {@code index}.
 */
public class DatasetShardGenerator extends DatasetGenerator {

  /** The position of the {@code num_shards} argument, excluding the receiver. */
  private static final int NUM_SHARDS_PARAMETER_POSITION = 0;

  /** The keyword name of the {@code num_shards} argument. */
  private static final String NUM_SHARDS_PARAMETER_NAME = "num_shards";

  /** The position of the {@code index} argument, excluding the receiver. */
  private static final int INDEX_PARAMETER_POSITION = 1;

  /** The keyword name of the {@code index} argument. */
  private static final String INDEX_PARAMETER_NAME = "index";

  protected DatasetShardGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetShardGenerator(CGNode node) {
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
   * @implNote For each receiver size {@code s}, shard count {@code n} and index {@code i}, the
   *     elements at {@code i, i + n, ...} below {@code s}: {@code ceil((s - i) / n)}, or none when
   *     {@code i} is at least {@code s}. The sizes are unknown when the receiver's are, or an
   *     argument does not resolve to a valid shard.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    Set<Long> sizes = super.getDatasetSizes(builder);
    if (sizes.isEmpty()) return sizes;
    Set<Long> shards =
        getPossibleLongValues(
            this.getArgumentPointsToSet(
                builder, NUM_SHARDS_PARAMETER_POSITION, NUM_SHARDS_PARAMETER_NAME));
    Set<Long> indices =
        getPossibleLongValues(
            this.getArgumentPointsToSet(builder, INDEX_PARAMETER_POSITION, INDEX_PARAMETER_NAME));
    if (shards == null || shards.isEmpty() || indices == null || indices.isEmpty())
      return Collections.emptySet();
    Set<Long> ret = HashSetFactory.make();
    for (Long size : sizes)
      for (Long shard : shards)
        for (Long index : indices) {
          // TensorFlow rejects a shard outside [0, num_shards); its size is then not known here.
          if (shard <= 0 || index < 0 || index >= shard) return Collections.emptySet();
          ret.add(index >= size ? 0L : (size - index + shard - 1) / shard);
        }
    return ret;
  }
}
