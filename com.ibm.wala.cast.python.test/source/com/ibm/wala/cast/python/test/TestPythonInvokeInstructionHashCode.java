package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.shrike.shrikeBT.IInvokeInstruction.Dispatch;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.Pair;
import java.util.HashMap;
import java.util.HashSet;
import org.junit.Test;

/**
 * Regression guards for <a href="https://github.com/wala/ML/issues/478">wala/ML#478</a>: {@link
 * PythonInvokeInstruction#hashCode} must be consistent with the inherited {@code final} {@code
 * SSAInstruction.equals}, which compares only the instruction index.
 */
public class TestPythonInvokeInstructionHashCode {

  private static CallSiteReference site(int pc) {
    MethodReference m =
        MethodReference.findOrCreate(
            PythonTypes.CodeBody, "do", "()Lcom/ibm/wala/cast/python/CodeBody;");
    return CallSiteReference.make(pc, m, Dispatch.VIRTUAL);
  }

  private static PythonInvokeInstruction make(int iindex, int[] positional) {
    @SuppressWarnings({"unchecked", "rawtypes"})
    Pair<String, Integer>[] kw = new Pair[0];
    return new PythonInvokeInstruction(iindex, 100, 101, site(42), positional, kw);
  }

  @Test
  public void equalInstructionsHashEqually() {
    PythonInvokeInstruction a = make(7, new int[] {1, 2, 3});
    PythonInvokeInstruction b = make(7, new int[] {4, 5});
    assertEquals(a, b);
    assertEquals("equal objects must have equal hash codes", a.hashCode(), b.hashCode());
  }

  @Test
  public void equalInstructionsDedupInHashSet() {
    PythonInvokeInstruction a = make(7, new int[] {1, 2, 3});
    PythonInvokeInstruction b = make(7, new int[] {4, 5});
    HashSet<PythonInvokeInstruction> set = new HashSet<>();
    set.add(a);
    set.add(b);
    assertEquals(1, set.size());
    assertTrue(set.contains(a));
    assertTrue(set.contains(b));
  }

  @Test
  public void equalInstructionsRoundTripThroughHashMap() {
    PythonInvokeInstruction a = make(7, new int[] {1, 2, 3});
    PythonInvokeInstruction b = make(7, new int[] {4, 5});
    HashMap<PythonInvokeInstruction, String> map = new HashMap<>();
    map.put(a, "found");
    assertEquals("found", map.get(b));
  }
}
