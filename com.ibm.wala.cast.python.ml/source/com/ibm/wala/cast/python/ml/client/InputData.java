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
 * <p>TODO(<a href="https://github.com/wala/ML/issues/470">wala/ML#470</a>): orphan flagged by the
 * dispatch-coverage meta-test — this class is not constructed from either dispatch table. Likely
 * superseded by {@link MnistInputData} (which is the actively-dispatched MNIST generator). Decide
 * to wire or delete; if delete, also drop {@link DispatchExempt} below.
 */
@DispatchExempt(
    "Orphan flagged by dispatch-coverage meta-test (wala/ML#470). Likely superseded by"
        + " MnistInputData; pending decide-to-wire-or-delete.")
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
