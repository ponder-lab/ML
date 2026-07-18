package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.SSAInstruction;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * Tests that a starred (unpacked) positional call argument (e.g. the {@code *rest} in {@code f(a,
 * *rest, b)}) is recorded on the {@link PythonInvokeInstruction} so a positional-alignment consumer
 * can degrade past it rather than mis-align every argument after the unpack. See <a
 * href="https://github.com/wala/ML/issues/751">wala/ML#751</a>.
 */
public class TestStarArgAlignment extends TestPythonMLCallGraphShape {

  /**
   * The {@code combine(tf.constant(...), *rest, i)} call in {@code tf751_star_align.py} has three
   * syntactic positional arguments after the callable: the tensor, the {@code *rest} unpack, and
   * the trailing {@code i}. The invoke therefore carries four positional slots (the callable plus
   * those three), and slot 2 (the {@code *rest}) is the sole starred slot. Before wala/ML#751 the
   * unpack was indistinguishable from an ordinary argument.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testStarredSlotRecorded() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf751_star_align.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    List<PythonInvokeInstruction> starredCalls = new ArrayList<>();
    for (CGNode node : CG) {
      if (!node.getMethod().getSignature().contains(".driver.")) continue;
      if (node.getIR() == null) continue;
      for (SSAInstruction inst : node.getIR().getInstructions())
        if (inst instanceof PythonInvokeInstruction) {
          PythonInvokeInstruction call = (PythonInvokeInstruction) inst;
          if (call.getStarredPositions().length > 0) starredCalls.add(call);
        }
    }

    assertEquals(
        "Exactly one call in `driver` unpacks a starred argument.", 1, starredCalls.size());
    PythonInvokeInstruction combineCall = starredCalls.get(0);
    assertEquals(
        "The `combine(tensor, *rest, i)` call has four positional slots.",
        4,
        combineCall.getNumberOfPositionalParameters());
    assertArrayEquals(
        "The `*rest` unpack is the sole starred slot, at index 2.",
        new int[] {2},
        combineCall.getStarredPositions());
    assertEquals(
        "The first starred slot bounds where positional alignment becomes unreliable.",
        2,
        combineCall.firstStarredPosition());
    assertTrue(combineCall.isPositionalSlotStarred(2));
  }

  /**
   * An ordinary call with no unpack records no starred slots, so the marker does not fire
   * spuriously and existing positional alignment is unaffected.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testNoStarredSlotOnOrdinaryCall() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf751_star_align.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    boolean sawOrdinaryCall = false;
    for (CGNode node : CG) {
      if (node.getIR() == null) continue;
      for (SSAInstruction inst : node.getIR().getInstructions())
        if (inst instanceof PythonInvokeInstruction) {
          PythonInvokeInstruction call = (PythonInvokeInstruction) inst;
          // `combine`'s body calls `tf.reduce_sum(a)`, an ordinary unstarred call.
          if (node.getMethod().getSignature().contains(".combine.")) {
            sawOrdinaryCall = true;
            assertEquals(
                "An ordinary call records no starred slots.", 0, call.getStarredPositions().length);
            assertEquals(-1, call.firstStarredPosition());
          }
        }
    }
    assertTrue("The reproduction must exercise `combine`'s ordinary calls.", sawOrdinaryCall);
  }

  /**
   * Captured gap for the argument-<em>binding</em> half of wala/ML#751. Exposing the starred slot
   * (above) lets a consumer distrust the alignment, but the analysis still binds the arguments
   * themselves incorrectly: in {@code combine(tf.constant(...), *rest, i)} the whole {@code rest}
   * sequence floods {@code combine}'s first covered parameter {@code b} instead of spreading across
   * {@code b} and {@code c}, and the trailing {@code i} never reaches {@code scale}. This is the
   * behavior the maintainer's points-to dump on wala/ML#751 records. Fixing it requires degrading
   * the actual-to-formal binding past a starred slot in <em>both</em> the trampoline forwarding and
   * the direct function-call dispatch (a plain function call like {@code combine} does not go
   * through the trampoline), so it is tracked separately from the marker exposure.
   *
   * <p>This test pins the current (incorrect) binding so it becomes a positive regression guard
   * once the binding degrades: {@code combine}'s {@code b} parameter (vn=3) currently points to the
   * {@code rest} list. When the binding is fixed, {@code b} will instead be unbound (empty
   * points-to set) because an unpack of statically-unknown length cannot be aligned to a specific
   * parameter.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/751">wala/ML#751</a>): flip this assertion
   * to expect an empty points-to set for {@code b} once the binding degrades past the unpack.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBindingFloodsFirstParameterPastUnpackCapturedGap() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf751_star_align.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

    boolean checkedCombine = false;
    for (CGNode node : CG) {
      if (!node.getMethod().getSignature().contains(".combine.")) continue;
      if (node.getIR() == null) continue;
      // combine.do()'s parameters are (callable, a, b, c, scale); `b` is vn=3.
      int b = node.getIR().getSymbolTable().getParameter(2);
      PointerKey bKey = pa.getHeapModel().getPointerKeyForLocal(node, b);
      assertFalse(
          "Captured gap for wala/ML#751: the `*rest` unpack floods `combine`'s `b` parameter"
              + " instead of leaving it unbound. Flip to expect an empty points-to set when the"
              + " actual-to-formal binding degrades past a starred slot.",
          pa.getPointsToSet(bKey).isEmpty());
      checkedCombine = true;
    }
    assertTrue("The reproduction must reach `combine`.", checkedCombine);
  }
}
