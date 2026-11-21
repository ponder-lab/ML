package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.util.collections.Pair;
import java.util.BitSet;
import org.junit.Test;

/**
 * The block-feasibility verdict is memoized per builder (<a
 * href="https://github.com/wala/ML/issues/1132">wala/ML#1132</a>). The analysis's whole-program
 * passes each ask about every block of every node, and each unmemoized ask walks the node's blocks.
 */
public class BlockFeasibilityMemoTest extends TestPythonMLCallGraphShape {

  /**
   * Every block's memoized verdict is the unmemoized walk's, one walk serves every block of a node,
   * and a second pass over every block makes no walk. The fixture's mode dispatch gives decidably
   * dead blocks, so the verdicts compared include {@link Boolean#FALSE}.
   */
  @Test
  public void testVerdictsAreTheWalksAndAreComputedOnce() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_mode_dispatch.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    try {
      int nodes = 0;
      int dead = 0;
      for (CGNode node : cg) {
        IR ir = node.getIR();
        BitSet walked = TensorGenerator.walkBlockFeasibility(builder, node);
        for (ISSABasicBlock block : ir.getControlFlowGraph())
          assertEquals(
              walked.get(block.getNumber()) ? Boolean.FALSE : null,
              TensorGenerator.computeBlockFeasibility(builder, node, block));
        nodes++;
        dead += walked.cardinality();
      }
      assertNotEquals("The fixture has no decidably dead block.", 0, dead);
      assertEquals(nodes, TensorGenerator.BLOCK_FEASIBILITIES.get(builder).walks);

      for (CGNode node : cg) {
        for (ISSABasicBlock block : node.getIR().getControlFlowGraph())
          TensorGenerator.computeBlockFeasibility(builder, node, block);
      }
      assertEquals(nodes, TensorGenerator.BLOCK_FEASIBILITIES.get(builder).walks);
    } finally {
      TensorGenerator.clearCaches(builder);
    }
    assertFalse(TensorGenerator.BLOCK_FEASIBILITIES.containsKey(builder));
  }

  /**
   * Under a phi fold, every block's verdict is the walk's, and the memo is neither read nor
   * written, so no entry exists for the builder afterwards. The frame on the stack is of no phi,
   * since no value number is 0, so the walk's folds answer as they do outside a fold.
   */
  @Test
  public void testVerdictsUnderAPhiFoldBypassTheMemo() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_mode_dispatch.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Pair<CGNode, Integer> frame = Pair.make(cg.getFakeRootNode(), 0);
    TensorGenerator.PHI_FOLD_ON_STACK.get().add(frame);
    try {
      int dead = 0;
      for (CGNode node : cg) {
        IR ir = node.getIR();
        BitSet walked = TensorGenerator.walkBlockFeasibility(builder, node);
        for (ISSABasicBlock block : ir.getControlFlowGraph())
          assertEquals(
              walked.get(block.getNumber()) ? Boolean.FALSE : null,
              TensorGenerator.computeBlockFeasibility(builder, node, block));
        dead += walked.cardinality();
      }
      assertNotEquals("The fixture has no decidably dead block.", 0, dead);
      assertFalse(TensorGenerator.BLOCK_FEASIBILITIES.containsKey(builder));
    } finally {
      TensorGenerator.PHI_FOLD_ON_STACK.get().remove(frame);
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * A walk during which a fold declined on an interim read is not memoized, so the next ask about
   * the node walks it again. With a resolver installed and no query resolved, the first walk of one
   * of the fixture's nodes has such a decline, and the walk that the second ask makes has none, so
   * it is memoized.
   */
  @Test
  public void testAWalkWithAnInterimDeclineIsNotMemoized() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_reshape_round_trip.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    WorklistTypeResolver.install(builder);
    try {
      int declined = 0;
      for (CGNode node : cg) {
        ISSABasicBlock entry = node.getIR().getControlFlowGraph().entry();
        TensorGenerator.computeBlockFeasibility(builder, node, entry);
        TensorGenerator.BlockFeasibilities feasibilities =
            TensorGenerator.BLOCK_FEASIBILITIES.get(builder);
        if (feasibilities.deadBlocks.containsKey(node)) continue;
        declined++;
        int walks = feasibilities.walks;
        TensorGenerator.computeBlockFeasibility(builder, node, entry);
        assertEquals(walks + 1, feasibilities.walks);
        assertNotNull(feasibilities.deadBlocks.get(node));
      }
      assertNotEquals("The fixture has no walk with an interim decline.", 0, declined);
    } finally {
      WorklistTypeResolver.uninstall(builder, false);
      TensorGenerator.clearCaches(builder);
    }
  }
}
