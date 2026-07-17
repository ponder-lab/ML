/*
 * Copyright (c) 2018 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ssa;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInstructionFactory;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class PythonInvokeInstruction extends SSAAbstractInvokeInstruction {
  private final int result;
  private final int[] positionalParams;
  private final Pair<String, Integer>[] keywordParams;

  /**
   * Indices into {@link #positionalParams} whose call-site argument was a starred (unpacked)
   * expression (e.g. the {@code *rest} in {@code f(a, *rest, b)}). A starred slot collapses an
   * unpacked sequence of statically-unknown length into a single positional slot, so a consumer
   * that aligns positional slots to callee parameters by index cannot trust the alignment of any
   * slot at or after a starred one. Empty for calls with no unpack (the common case). See <a
   * href="https://github.com/wala/ML/issues/751">wala/ML#751</a>.
   */
  private final int[] starredPositions;

  public PythonInvokeInstruction(
      int iindex,
      int result,
      int exception,
      CallSiteReference site,
      int[] positionalParams,
      Pair<String, Integer>[] keywordParams) {
    this(iindex, result, exception, site, positionalParams, keywordParams, EMPTY_STARRED);
  }

  public PythonInvokeInstruction(
      int iindex,
      int result,
      int exception,
      CallSiteReference site,
      int[] positionalParams,
      Pair<String, Integer>[] keywordParams,
      int[] starredPositions) {
    super(iindex, exception, site);
    this.positionalParams = positionalParams;
    this.keywordParams = keywordParams;
    this.starredPositions = starredPositions;
    this.result = result;
  }

  private static final int[] EMPTY_STARRED = new int[0];

  @Override
  public int getNumberOfPositionalParameters() {
    return positionalParams.length;
  }

  public int getNumberOfKeywordParameters() {
    return keywordParams.length;
  }

  /**
   * The positional-slot indices whose call-site argument was a starred (unpacked) expression.
   *
   * @return the starred positional-slot indices, empty if the call has no unpack.
   */
  public int[] getStarredPositions() {
    return starredPositions;
  }

  /**
   * Whether the given positional slot's call-site argument was a starred (unpacked) expression.
   *
   * @param slot The positional-slot index to test.
   * @return {@code true} iff the slot was a starred argument.
   */
  public boolean isPositionalSlotStarred(int slot) {
    for (int s : starredPositions) if (s == slot) return true;
    return false;
  }

  /**
   * The smallest starred positional-slot index, at or after which positional-to-parameter alignment
   * is unreliable (an unpack of statically-unknown length precedes those slots). See <a
   * href="https://github.com/wala/ML/issues/751">wala/ML#751</a>.
   *
   * @return the first starred slot index, or {@code -1} if the call has no unpack.
   */
  public int firstStarredPosition() {
    int min = -1;
    for (int s : starredPositions) if (min == -1 || s < min) min = s;
    return min;
  }

  public int getNumberOfTotalParameters() {
    return positionalParams.length + keywordParams.length;
  }

  @Override
  public int getNumberOfUses() {
    return positionalParams.length + keywordParams.length;
  }

  public List<String> getKeywords() {
    List<String> names = new LinkedList<String>();
    for (Pair<String, ?> a : keywordParams) {
      names.add(a.fst);
    }
    return names;
  }

  public int getUse(String keyword) {
    for (int i = 0; i < keywordParams.length; i++) {
      if (keywordParams[i].fst.equals(keyword)) {
        return keywordParams[i].snd;
      }
    }

    return -1;
  }

  @Override
  public int getUse(int j) throws UnsupportedOperationException {
    if (j < positionalParams.length) {
      return positionalParams[j];
    } else {
      assert j < getNumberOfTotalParameters();
      return keywordParams[j - positionalParams.length].snd;
    }
  }

  @Override
  public int getNumberOfReturnValues() {
    return 1;
  }

  @Override
  public int getReturnValue(int i) {
    assert i == 0;
    return result;
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  @Override
  public SSAInstruction copyForSSA(SSAInstructionFactory insts, int[] defs, int[] uses) {
    int nr = defs == null || defs.length == 0 ? result : defs[0];
    int ne = defs == null || defs.length == 0 ? exception : defs[1];

    int[] newpos = positionalParams;
    Pair<String, Integer>[] newkey = keywordParams;
    if (uses != null && uses.length > 0) {
      int j = 0;
      newpos = new int[positionalParams.length];
      for (int i = 0; i < positionalParams.length; i++, j++) {
        newpos[i] = uses[j];
      }
      newkey = new Pair[keywordParams.length];
      for (int i = 0; i < keywordParams.length; i++, j++) {
        newkey[i] = Pair.make(keywordParams[i].fst, uses[j]);
      }
    }

    return new PythonInvokeInstruction(iIndex(), nr, ne, site, newpos, newkey, starredPositions);
  }

  @Override
  public void visit(IVisitor v) {
    ((PythonInstructionVisitor) v).visitPythonInvoke(this);
  }

  /**
   * Hashes purely on {@code iIndex()}. {@link SSAInstruction#equals} is {@code final} and compares
   * only the instruction index, so any wider hash (the prior {@code getCallSite().hashCode() *
   * result} included a value number) would let two instances be {@code equals}-equal but hash to
   * different buckets — breaking the {@code HashMap}/{@code HashSet} invariant. See <a
   * href="https://github.com/wala/ML/issues/478">wala/ML#478</a>.
   *
   * @return a hash code consistent with {@link SSAInstruction#equals}.
   */
  @Override
  public int hashCode() {
    return iIndex();
  }

  @Override
  public Collection<TypeReference> getExceptionTypes() {
    return Collections.singleton(PythonTypes.Exception);
  }

  @Override
  public String toString(SymbolTable symbolTable) {
    String s = "";
    if (keywordParams != null) {
      for (Pair<String, Integer> kp : keywordParams) {
        s = s + " " + kp.fst + ":" + kp.snd;
      }
    }
    return super.toString(symbolTable) + s;
  }
}
