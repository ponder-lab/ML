package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for the window {@code tf.data.Dataset.group_by_window} hands its {@code reduce_func}
 * (wala/ML#1214). Its element is the receiver's, as {@code filter}'s is, but it holds only the
 * receiver's elements that share a key, up to the window's size, so how many it holds is not the
 * receiver's count.
 */
public class DatasetGroupWindowGenerator extends DatasetGenerator {

  protected DatasetGroupWindowGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetGroupWindowGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown. A window holds the elements sharing a key, at most the window's size of
   *     them, and the last window of a key may hold fewer.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
