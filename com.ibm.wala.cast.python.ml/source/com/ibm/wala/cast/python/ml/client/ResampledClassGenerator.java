package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for the class tensor of an element of {@code tf.data.Dataset.rejection_resample}, an
 * int32 scalar (wala/ML#1214).
 */
@DispatchExempt(
    "Constructed only by delegation: a resampled dataset's generator makes it for the class of an"
        + " element that is not copied ({@link"
        + " DatasetRejectionResampleGenerator#getIndexGenerator}); no dispatch-table type names it"
        + " (wala/ML#1214).")
public class ResampledClassGenerator extends EnumerateIndexGenerator {

  protected ResampledClassGenerator(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(DType.INT32);
  }
}
