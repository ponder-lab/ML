package com.ibm.wala.cast.python.ssa;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction.IterationSource;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.graph.dominators.Dominators;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The loops of one {@link IR} that the front end lowered from a {@code for} statement or a
 * comprehension's generator, indexed by the {@link EachElementGetInstruction}s that step them
 * (wala/ML#1133).
 *
 * <p>Each such loop starts with the {@code iter} call over its iterable, and its element gets come
 * after that call and before any {@code iter} or {@code next} call of a loop nested in it, so the
 * nearest iteration-protocol call that dominates an element get is its loop's {@code iter} call.
 * Matching on the iterable's value number instead would conflate two loops over one variable.
 */
public final class IterationSites {

  /** The {@code iter} call of the loop each element get steps. */
  private final Map<EachElementGetInstruction, PythonInvokeInstruction> iteratorCalls =
      HashMapFactory.make();

  /**
   * Indexes the loops of the given IR.
   *
   * @param ir The IR whose loops to index.
   */
  public IterationSites(IR ir) {
    SSACFG cfg = ir.getControlFlowGraph();
    List<PythonInvokeInstruction> protocolCalls = new ArrayList<>();
    List<EachElementGetInstruction> gets = new ArrayList<>();
    for (SSAInstruction instruction : ir.getInstructions())
      if (instruction instanceof PythonInvokeInstruction call && call.isIterationProtocol())
        protocolCalls.add(call);
      else if (instruction instanceof EachElementGetInstruction get) gets.add(get);
    if (protocolCalls.isEmpty() || gets.isEmpty()) return;

    Dominators<ISSABasicBlock> dominators = Dominators.make(cfg, cfg.entry());
    for (EachElementGetInstruction get : gets) {
      ISSABasicBlock getBlock = cfg.getBlockForInstruction(get.iIndex());
      PythonInvokeInstruction nearest = null;
      ISSABasicBlock nearestBlock = null;
      for (PythonInvokeInstruction call : protocolCalls) {
        ISSABasicBlock callBlock = cfg.getBlockForInstruction(call.iIndex());
        boolean dominates =
            callBlock.equals(getBlock)
                ? call.iIndex() < get.iIndex()
                : dominators.isDominatedBy(getBlock, callBlock);
        if (!dominates) continue;
        // The dominators of a block form a chain, so the nearest is the one the others dominate.
        if (nearest == null
            || (callBlock.equals(nearestBlock)
                ? call.iIndex() > nearest.iIndex()
                : dominators.isDominatedBy(callBlock, nearestBlock))) {
          nearest = call;
          nearestBlock = callBlock;
        }
      }
      if (nearest != null) iteratorCalls.put(get, nearest);
    }
  }

  /**
   * The {@code iter} call of the loop the given element get steps.
   *
   * @param get An element get of the indexed IR.
   * @return The call, or {@code null} when the front end did not lower the get from a {@code for}
   *     statement or a comprehension's generator.
   */
  public PythonInvokeInstruction getIteratorCall(EachElementGetInstruction get) {
    return iteratorCalls.get(get);
  }

  /**
   * The source construct of the loop the given element get steps.
   *
   * @param get An element get of the indexed IR.
   * @return The construct, or {@code null} when the front end did not lower the get from a {@code
   *     for} statement or a comprehension's generator.
   */
  public IterationSource getIterationSource(EachElementGetInstruction get) {
    PythonInvokeInstruction call = getIteratorCall(get);
    return call == null ? null : call.getIterationSource();
  }

  /**
   * The value number of the iterable the loop the given element get steps iterates.
   *
   * @param get An element get of the indexed IR.
   * @return The iterable, or {@code -1} when the front end did not lower the get from a {@code for}
   *     statement or a comprehension's generator.
   */
  public int getIterable(EachElementGetInstruction get) {
    PythonInvokeInstruction call = getIteratorCall(get);
    // The `iter` call's positional parameters are the builtin, then the iterable.
    return call == null ? -1 : call.getUse(1);
  }
}
