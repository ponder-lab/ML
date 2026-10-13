package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAPhiInstruction;
import java.util.Iterator;
import org.junit.Test;

/**
 * A phi is found in an IR by the value it defines, not by identity (<a
 * href="https://github.com/wala/ML/issues/1040">wala/ML#1040</a>). A node's {@code DefUse} can be
 * built over another IR of the same node than the one its blocks are read from, so the phi a
 * definition lookup returns may be a distinct instance from any phi of those blocks; matched by
 * identity, the lookup missed, and whether it missed depended on which IR each cache held.
 */
public class PhiBlockTest extends TestPythonMLCallGraphShape {

  @Test
  public void testPhiFoundByDefinitionFromAnotherInstance() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_none_conditional.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    CGNode statement = null;
    for (CGNode node : cg)
      if (node.getMethod().getSignature().contains("tf2_test_none_conditional.py.statement.do"))
        statement = node;
    assertNotNull("The function `statement` has no node.", statement);

    IR ir = statement.getIR();
    SSAPhiInstruction phi = null;
    ISSABasicBlock block = null;
    for (ISSABasicBlock candidate : ir.getControlFlowGraph())
      for (Iterator<SSAPhiInstruction> it = candidate.iteratePhis(); it.hasNext(); ) {
        phi = it.next();
        block = candidate;
      }
    assertNotNull("The function `statement` has no phi.", phi);

    int[] uses = new int[phi.getNumberOfUses()];
    for (int i = 0; i < uses.length; i++) uses[i] = phi.getUse(i);
    SSAPhiInstruction copy = new SSAPhiInstruction(phi.iIndex(), phi.getDef(), uses);
    assertNotSame(phi, copy);

    assertSame(block, PythonTensorAnalysisEngine.phiBlock(ir, copy));

    // A value no phi defines, such as the function's first parameter, has no phi block.
    assertNull(
        PythonTensorAnalysisEngine.phiBlock(ir, new SSAPhiInstruction(phi.iIndex(), 1, uses)));
  }
}
