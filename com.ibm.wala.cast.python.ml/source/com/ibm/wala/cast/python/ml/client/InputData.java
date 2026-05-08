package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for MNIST input data tensors. These tensors have the shape defined by
 * TensorType.mnistInput().
 *
 * <p>TODO(<a href="https://github.com/wala/ML/issues/483">wala/ML#483</a>): orphan flagged by the
 * dispatch-coverage meta-test (wala/ML#470). Recommended action is delete — likely superseded by
 * {@link MnistInputData} (the actively-dispatched MNIST generator). When deleting, also drop {@link
 * DispatchExempt} below.
 */
@DispatchExempt(
    "Orphan flagged by dispatch-coverage meta-test (wala/ML#470). Recommended delete per"
        + " wala/ML#483; likely superseded by MnistInputData.")
public class InputData extends TensorGenerator {
  public InputData(PointsToSetVariable source) {
    super(source);
  }

  public InputData(CGNode node) {
    super(node);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(TensorType.mnistInput().getDims());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.FLOAT32);
  }
}
