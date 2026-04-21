package com.ibm.wala.cast.python.ssa;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInstructionFactory;
import com.ibm.wala.types.TypeReference;
import java.util.Collection;
import java.util.Collections;

/**
 * Python-specific binary-op instruction. Exists so analyses can hook Python binops without
 * pattern-matching {@link SSABinaryOpInstruction} against the surrounding language. Dispatches to
 * {@link PythonInstructionVisitor#visitPythonBinaryOp(PythonBinaryOpInstruction)} instead of the
 * generic {@code visitBinaryOp}; this lets the call-graph builder synthesise a per-instruction
 * allocation site for the result (wala/ML#398), which the pointer analysis needs so downstream
 * consumers see a non-empty PTS for binop-produced tensors.
 */
public class PythonBinaryOpInstruction extends SSABinaryOpInstruction {

  public PythonBinaryOpInstruction(
      int iindex,
      IBinaryOpInstruction.IOperator operator,
      int result,
      int val1,
      int val2,
      boolean mayBeInteger) {
    super(iindex, operator, result, val1, val2, mayBeInteger);
  }

  @Override
  public SSAInstruction copyForSSA(SSAInstructionFactory insts, int[] defs, int[] uses) {
    return new PythonBinaryOpInstruction(
        iIndex(),
        getOperator(),
        defs == null || defs.length == 0 ? getDef(0) : defs[0],
        uses == null ? getUse(0) : uses[0],
        uses == null ? getUse(1) : uses[1],
        mayBeIntegerOp());
  }

  @Override
  public Collection<TypeReference> getExceptionTypes() {
    return Collections.singleton(PythonTypes.Exception);
  }

  @Override
  public void visit(IVisitor v) {
    ((PythonInstructionVisitor) v).visitPythonBinaryOp(this);
  }
}
