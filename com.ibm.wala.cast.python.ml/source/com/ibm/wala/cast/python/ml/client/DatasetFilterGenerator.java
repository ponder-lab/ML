package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.filter} (wala/ML#1237). The element is
 * the receiver's, but the predicate may drop any of the receiver's elements, so how many the
 * dataset has is not counted.
 */
public class DatasetFilterGenerator extends DatasetGenerator {

  public DatasetFilterGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetFilterGenerator(CGNode node) {
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
   * @implNote A filter of a dataset that never ends never ends either: it yields each element the
   *     predicate keeps, or waits for one without end, so a batch of it has no last partial batch.
   */
  @Override
  public boolean passesReceiverThrough() {
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown. The predicate may keep any number of the receiver's elements, up to all.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
