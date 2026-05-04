package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.math.argmax}. Output dtype is fixed at {@link DType#INT64} (the TF
 * default; the {@code output_type=tf.int32} alternative isn't currently surfaced through {@code
 * tensorflow.xml}, so the generator emits {@code int64} unconditionally). Output shape is left at ⊤
 * for now; computing the precise shape (input.shape with the {@code axis} dimension removed)
 * regresses tests such as {@code testNeuralNetwork*}, where the per-context shape union for {@code
 * y_true} (e.g. {@code [256]} train vs. {@code [10000]} test) cross-products through {@code
 * ElementWiseOperation}'s strict broadcast check. Shape precision can be added once the EWO union
 * behaviour is addressed (wala/ML#462). See wala/ML#449 (Tier 6).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/argmax">tf.math.argmax</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Argmax extends ReduceMean {

  public Argmax(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.INT64);
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.INT64);
  }
}
