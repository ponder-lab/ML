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

import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInstructionFactory;
import com.ibm.wala.types.TypeReference;
import java.util.Collection;
import java.util.Collections;

public class PythonPropertyRead extends AstPropertyRead {
  /**
   * Whether the read came from subscript syntax, {@code y[k]}, rather than attribute syntax, {@code
   * y.k}. Both lower to this instruction, and a string-constant subscript key makes the two
   * otherwise identical (wala/ML#1100).
   */
  private final boolean subscript;

  /**
   * Whether the read came from destructuring, {@code a, b = y} or a {@code for a, b in ys} target,
   * rather than from attribute or subscript syntax. A destructuring reads each target out of its
   * value, which at run time iterates the value, and lowers to this instruction. Its member is the
   * target's position, an integer constant, except under a starred target, {@code a, *rest = y},
   * where it is the order-free contents field or the starred marker (wala/ML#1117).
   */
  private final boolean unpacking;

  public PythonPropertyRead(int iindex, int result, int objectRef, int memberRef) {
    this(iindex, result, objectRef, memberRef, false);
  }

  /**
   * A property read that records the syntax it came from (wala/ML#1100).
   *
   * @param iindex The instruction index.
   * @param result The read value's value number.
   * @param objectRef The receiver's value number.
   * @param memberRef The member's value number: an attribute's name, or a subscript's key.
   * @param subscript Whether the read came from subscript syntax rather than attribute syntax.
   */
  public PythonPropertyRead(
      int iindex, int result, int objectRef, int memberRef, boolean subscript) {
    this(iindex, result, objectRef, memberRef, subscript, false);
  }

  private PythonPropertyRead(
      int iindex, int result, int objectRef, int memberRef, boolean subscript, boolean unpacking) {
    super(iindex, result, objectRef, memberRef);
    assert !(subscript && unpacking) : "A read comes from one syntax.";
    this.subscript = subscript;
    this.unpacking = unpacking;
  }

  /**
   * A property read that destructuring makes: it reads one target out of the destructured value
   * (wala/ML#1117).
   *
   * @param iindex The instruction index.
   * @param result The target's value number.
   * @param objectRef The destructured value's value number.
   * @param memberRef The member's value number: the target's position, an integer constant, or
   *     under a starred target the contents field or the starred marker.
   * @return The read, which {@link #isUnpacking()}.
   */
  public static PythonPropertyRead unpacking(int iindex, int result, int objectRef, int memberRef) {
    return new PythonPropertyRead(iindex, result, objectRef, memberRef, false, true);
  }

  /**
   * Whether this read came from subscript syntax, {@code y[k]}, rather than attribute syntax,
   * {@code y.k} (wala/ML#1100).
   *
   * @return {@code true} iff the read is a subscript read.
   */
  public boolean isSubscript() {
    return subscript;
  }

  /**
   * Whether this read came from destructuring, {@code a, b = y} or a {@code for a, b in ys} target,
   * rather than from attribute or subscript syntax (wala/ML#1117). Such a read iterates its
   * receiver at run time, where a subscript calls {@code __getitem__}. It is never also a {@link
   * #isSubscript() subscript}.
   *
   * @return {@code true} iff the read is a destructuring read.
   */
  public boolean isUnpacking() {
    return unpacking;
  }

  @Override
  public SSAInstruction copyForSSA(SSAInstructionFactory insts, int[] defs, int[] uses) {
    return new PythonPropertyRead(
        iIndex(),
        defs == null ? getDef() : defs[0],
        uses == null ? getObjectRef() : uses[0],
        uses == null ? getMemberRef() : uses[1],
        subscript,
        unpacking);
  }

  /* (non-Javadoc)
   * @see com.ibm.domo.ssa.Instruction#getExceptionTypes()
   */
  @Override
  public Collection<TypeReference> getExceptionTypes() {
    return Collections.singleton(PythonTypes.Root);
  }
}
