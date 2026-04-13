package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Modeling of the NumPy astype() operation. This operation preserves the shape of the receiver
 * tensor.
 */
public class AstypeOperation extends TensorGenerator {
  private static final Logger LOGGER = Logger.getLogger(AstypeOperation.class.getName());

  public AstypeOperation(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int receiverVn = getReceiverVn();
    if (receiverVn > 0) {
      Set<List<Dimension<?>>> shapes = getShapes(builder, getNode(), receiverVn);
      if (shapes != null && !shapes.isEmpty()) {
        LOGGER.info("AstypeOperation returning shapes: " + shapes);
        return shapes;
      }
    }
    return null;
  }

  private int getReceiverVn() {
    com.ibm.wala.cast.python.ssa.PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int funcVn = call.getUse(0);
      SSAInstruction funcDef = getNode().getDU().getDef(funcVn);
      if (funcDef instanceof PythonPropertyRead) {
        return ((PythonPropertyRead) funcDef).getObjectRef();
      }
    }
    return getArgumentValueNumber(RECEIVER_PARAMETER_POSITION);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int arg0Vn = getArgumentValueNumber(0);
    if (arg0Vn > 0) {
      Set<DType> dTypes = getDTypes(builder, arg0Vn);
      if (!dTypes.isEmpty()) {
        return dTypes;
      }
    }
    return Set.of(DType.FLOAT32);
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
}
