package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.boolean_mask}. Output dtype is inherited from the {@code tensor} input.
 * Output shape is left at ⊤ for now: the masked dimension's length is the number of {@code True}
 * entries in {@code mask}, a runtime quantity that static analysis cannot recover. The remaining
 * dimensions follow {@code tensor.shape} (offset by {@code axis}), so the result rank is generally
 * {@code tensor.rank - mask.rank + 1}, not always 1-D — but the dependence on the mask's runtime
 * truthiness still leaves the masked dimension undetermined. See wala/ML#449 (Tier 8).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/boolean_mask">tf.boolean_mask</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class BooleanMask extends PassThroughUnaryTensorGenerator {

  public BooleanMask(PointsToSetVariable source) {
    super(source);
  }

  public BooleanMask(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "tensor";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
