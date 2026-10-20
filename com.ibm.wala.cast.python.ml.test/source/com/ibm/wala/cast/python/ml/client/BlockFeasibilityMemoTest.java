package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import org.junit.Test;

/**
 * The block-feasibility verdict is memoized per builder (<a
 * href="https://github.com/wala/ML/issues/1132">wala/ML#1132</a>). The analysis's whole-program
 * passes each ask about every block of every node, and each unmemoized ask walked the block's
 * predecessors back to the entry.
 */
public class BlockFeasibilityMemoTest extends TestPythonMLCallGraphShape {

  /**
   * Every block's memoized verdict is the unmemoized walk's, and a second pass over every block
   * makes no walk. The fixture's mode dispatch gives decidably dead blocks, so the verdicts
   * compared include {@link Boolean#FALSE}.
   */
  @Test
  public void testVerdictsAreTheWalksAndAreComputedOnce() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_mode_dispatch.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    try {
      int blocks = 0;
      int dead = 0;
      for (CGNode node : cg) {
        IR ir = node.getIR();
        if (ir == null) continue;
        for (ISSABasicBlock block : ir.getControlFlowGraph()) {
          Boolean walked = TensorGenerator.walkBlockFeasibility(builder, node, block);
          assertEquals(walked, TensorGenerator.computeBlockFeasibility(builder, node, block));
          blocks++;
          if (Boolean.FALSE.equals(walked)) dead++;
        }
      }
      assertTrue("The fixture has no decidably dead block.", dead > 0);
      assertEquals(blocks, TensorGenerator.BLOCK_FEASIBILITIES.get(builder).walks);

      for (CGNode node : cg) {
        IR ir = node.getIR();
        if (ir == null) continue;
        for (ISSABasicBlock block : ir.getControlFlowGraph())
          TensorGenerator.computeBlockFeasibility(builder, node, block);
      }
      assertEquals(blocks, TensorGenerator.BLOCK_FEASIBILITIES.get(builder).walks);
    } finally {
      TensorGenerator.clearCaches(builder);
    }
    assertTrue(!TensorGenerator.BLOCK_FEASIBILITIES.containsKey(builder));
  }
}
