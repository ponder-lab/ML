/*
 * Copyright (c) 2002 - 2006 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ssa;

import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInstructionFactory;
import com.ibm.wala.types.TypeReference;
import java.util.Collection;
import java.util.Collections;

public class PythonPropertyWrite extends AstPropertyWrite {

  /**
   * Whether the write came from subscript syntax, {@code y[k] = v}, rather than attribute syntax,
   * {@code y.k = v}. Both lower to this instruction, and a string-constant subscript key makes the
   * two otherwise identical (wala/ML#1100).
   */
  private final boolean subscript;

  public PythonPropertyWrite(int iindex, int objectRef, int memberRef, int value) {
    this(iindex, objectRef, memberRef, value, false);
  }

  /**
   * A property write that records the syntax it came from (wala/ML#1100).
   *
   * @param iindex The instruction index.
   * @param objectRef The receiver's value number.
   * @param memberRef The member's value number: an attribute's name, or a subscript's key.
   * @param value The written value's value number.
   * @param subscript Whether the write came from subscript syntax rather than attribute syntax.
   */
  public PythonPropertyWrite(
      int iindex, int objectRef, int memberRef, int value, boolean subscript) {
    super(iindex, objectRef, memberRef, value);
    this.subscript = subscript;
  }

  /**
   * Whether this write came from subscript syntax, {@code y[k] = v}, rather than attribute syntax,
   * {@code y.k = v} (wala/ML#1100).
   *
   * @return {@code true} iff the write is a subscript store.
   */
  public boolean isSubscript() {
    return subscript;
  }

  @Override
  public SSAInstruction copyForSSA(SSAInstructionFactory insts, int[] defs, int[] uses) {
    return new PythonPropertyWrite(
        iIndex(),
        uses == null ? getObjectRef() : uses[0],
        uses == null ? getMemberRef() : uses[1],
        uses == null ? getValue() : uses[2],
        subscript);
  }

  /*
   * (non-Javadoc)
   * @see com.ibm.domo.ssa.Instruction#getExceptionTypes()
   */
  @Override
  public Collection<TypeReference> getExceptionTypes() {
    return Collections.singleton(PythonTypes.Root);
  }
}
