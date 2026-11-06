package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for datasets created by {@code tf.data.Dataset.rejection_resample} (wala/ML#1214).
 * The element is the pair {@code (class, element)}: the int32 scalar {@code class_func} returns at
 * position 0 and the receiver's element at position 1, as {@code enumerate}'s is with its index.
 */
public class DatasetRejectionResampleGenerator extends DatasetEnumerateGenerator {

  public DatasetRejectionResampleGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetRejectionResampleGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @return {@link DType#INT32}, which {@code class_func} must return.
   */
  @Override
  protected DType getIndexDType() {
    return DType.INT32;
  }

  @Override
  public TensorGenerator getIndexGenerator(PointsToSetVariable source) {
    return new ResampledClassGenerator(source);
  }
}
