package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.bucket_by_sequence_length}
 * (wala/ML#1214). The element is a batch of the receiver's, each element padded to the longest in
 * its batch. The batch size depends on the bucket, so it is not counted. {@code
 * element_length_func} is not called.
 */
public class DatasetBucketBySequenceLengthGenerator extends DatasetBatchGenerator {

  /** The position of the {@code padded_shapes} argument, excluding the receiver. */
  private static final int PADDED_SHAPES_PARAMETER_POSITION = 3;

  /** The keyword name of the {@code padded_shapes} argument. */
  private static final String PADDED_SHAPES_PARAMETER_NAME = "padded_shapes";

  /** The position of the {@code pad_to_bucket_boundary} argument, excluding the receiver. */
  private static final int PAD_TO_BUCKET_BOUNDARY_PARAMETER_POSITION = 5;

  /** The keyword name of the {@code pad_to_bucket_boundary} argument. */
  private static final String PAD_TO_BUCKET_BOUNDARY_PARAMETER_NAME = "pad_to_bucket_boundary";

  /** The position of the {@code drop_remainder} argument, excluding the receiver. */
  private static final int DROP_REMAINDER_PARAMETER_POSITION = 7;

  public DatasetBucketBySequenceLengthGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetBucketBySequenceLengthGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected int getDropRemainderParameterIndex() {
    return DROP_REMAINDER_PARAMETER_POSITION;
  }

  /**
   * Whether padding to the longest element in a batch leaves each element's shape as it is: only
   * when no {@code padded_shapes} is supplied, which declares the padded shape, and {@code
   * pad_to_bucket_boundary} is omitted or provably {@code False}, since padding to a bucket's
   * boundary changes the padded axis.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @return {@code true} iff the padded element has the receiver's element's shapes.
   */
  private boolean keepsElementShapes(PropagationCallGraphBuilder builder) {
    if (!Boolean.FALSE.equals(
        this.isArgumentSyntacticallySupplied(
            builder, PADDED_SHAPES_PARAMETER_POSITION, PADDED_SHAPES_PARAMETER_NAME))) return false;
    Boolean toBoundary =
        this.isArgumentSyntacticallySupplied(
            builder,
            PAD_TO_BUCKET_BOUNDARY_PARAMETER_POSITION,
            PAD_TO_BUCKET_BOUNDARY_PARAMETER_NAME);
    if (Boolean.FALSE.equals(toBoundary)) return true;
    if (toBoundary == null) return false;
    Set<Boolean> values =
        getPossibleBooleanValues(
            this.getArgumentPointsToSet(
                builder,
                PAD_TO_BUCKET_BOUNDARY_PARAMETER_POSITION,
                PAD_TO_BUCKET_BOUNDARY_PARAMETER_NAME));
    return values != null && values.equals(Collections.singleton(Boolean.FALSE));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Each shape behind a batch dimension of unknown size, where padding leaves it as it
   *     is, and unknown otherwise.
   */
  @Override
  protected Set<List<Dimension<?>>> applyBatching(
      Set<List<Dimension<?>>> inputShapes, PropagationCallGraphBuilder builder) {
    if (!this.keepsElementShapes(builder)) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      List<Dimension<?>> batched = new ArrayList<>();
      batched.add(new SymbolicDim("?"));
      batched.addAll(shape);
      ret.add(batched);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown: the number of batches depends on how the elements fall into the buckets.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
