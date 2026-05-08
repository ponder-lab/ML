package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code tf.strings.as_string}. Output shape is the input's shape (passthrough on
 * {@code input}); output dtype is fixed to {@link DType#STRING} regardless of the input's numeric
 * dtype.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/strings/as_string">tf.strings.as_string</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class AsString extends PassThroughUnaryTensorGenerator {

  public AsString(PointsToSetVariable source) {
    super(source);
  }

  public AsString(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input";
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.STRING);
  }
}
