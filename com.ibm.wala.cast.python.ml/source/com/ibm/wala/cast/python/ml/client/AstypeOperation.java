package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
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
    LOGGER.fine(
        () -> "AstypeOperation.getDefaultShapes: source=" + source + ", receiverVn=" + receiverVn);
    if (receiverVn > 0) {
      try {
        Set<List<Dimension<?>>> shapes = getShapes(builder, getNode(), receiverVn);
        LOGGER.fine(
            () ->
                "AstypeOperation.getDefaultShapes: shapes from receiverVn="
                    + receiverVn
                    + " -> "
                    + shapes);
        if (shapes != null && !shapes.isEmpty()) {
          return shapes;
        }
      } catch (IllegalArgumentException e) {
        // `getShapes` throws when the receiver's PTS is empty AND its PointerKey is implicit —
        // e.g., a chained `x.astype(int32).astype(float32)` where the inner call's return value
        // is a synthetic-method return (implicit PK, no materialised PTS). The multi-stage
        // helper in `TensorGenerator.getShapes(builder, node, vn)` skips the factory-recursion
        // branch for implicit keys and falls through to IAE. Catch and return `null` (⊤ unknown
        // shape) so dtype inference still proceeds and the result flows downstream as a tensor
        // instead of being dropped entirely. For the non-chained mnist case, the factory
        // recursion fires successfully via `MnistInputData`, so the shape is recovered and this
        // catch doesn't fire. See wala/ML#356, wala/WALA#1889.
        LOGGER.log(
            Level.FINE,
            "AstypeOperation.getDefaultShapes: receiver shape lookup failed for receiverVn="
                + receiverVn,
            e);
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
