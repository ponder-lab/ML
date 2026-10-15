package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SymbolTable;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * A key read of a comprehension-built dict queried outside an analysis run (<a
 * href="https://github.com/wala/ML/issues/1071">wala/ML#1071</a>).
 *
 * <p>Inside a run, the read re-derives the comprehension's written value under a binding that the
 * type resolver validates once it converges. Outside a run there is no resolver (wala/ML#753), so
 * nothing would validate it, and the read keeps the generator it had before: none.
 */
public class TestDictComprehensionReadOutsideAnalysis extends TestPythonMLCallGraphShape {

  /**
   * The factory, asked directly for the read of {@code sliced["audios"]} after only the call graph
   * is built, gives the read no generator.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testNoGeneratorOutsideAnalysis() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_dict_comprehension_controls.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    List<PointerKey> reads = new ArrayList<>();
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().contains(".two_keys.")) continue;
      SymbolTable st = node.getIR().getSymbolTable();
      for (SSAInstruction inst : node.getIR().getInstructions())
        if (inst instanceof PythonPropertyRead read
            && st.isStringConstant(read.getMemberRef())
            && "audios".equals(st.getStringValue(read.getMemberRef()))
            && node.getDU().getDef(read.getObjectRef()) instanceof SSANewInstruction)
          reads.add(
              builder
                  .getPointerAnalysis()
                  .getHeapModel()
                  .getPointerKeyForLocal(node, read.getDef()));
    }
    // The read is found, so the null below is the factory's answer for it.
    assertEquals("The comprehension read was not found.", 1, reads.size());
    assertNull(
        TensorGeneratorFactory.getGenerator(
            builder.getPropagationSystem().findOrCreatePointsToSet(reads.get(0)), builder));
  }
}
