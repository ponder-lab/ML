package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.SSAContextInterpreter;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

public class TestPrimitives extends TestJythonCallGraphShape {

  @Test
  public void testSource1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("prim1.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    // verifyGraphAssertions(CG, assertionsCalls1);
  }

  /**
   * A binary operation over two literals folds to its own value: a float to a {@link Double} rather
   * than a truncated integer. A comparison, and a float that overflows to infinity, stay unfolded
   * (<a href="https://github.com/wala/ML/issues/1143">wala/ML#1143</a>). The arguments of the calls
   * to {@code sink} in {@code fold1.py} are read in program order.
   */
  @Test
  public void testFloatFolding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("fold1.py");
    CGNode script = getNodes(CG, "script fold1.py").iterator().next();
    IR ir = script.getIR();
    SymbolTable symtab = ir.getSymbolTable();
    // Each call to `sink` has its own context, so its own node.
    Set<SSAAbstractInvokeInstruction> calls = new HashSet<>();
    for (CGNode sink : getNodes(CG, "script fold1.py/sink"))
      CG.getPossibleSites(script, sink)
          .forEachRemaining(site -> calls.addAll(List.of(ir.getCalls(site))));
    List<SSAAbstractInvokeInstruction> ordered = new ArrayList<>(calls);
    ordered.sort(Comparator.comparingInt(SSAAbstractInvokeInstruction::iIndex));
    List<Object> arguments = new ArrayList<>();
    for (SSAAbstractInvokeInstruction call : ordered) {
      int argument = call.getUse(1);
      arguments.add(symtab.isConstant(argument) ? symtab.getConstantValue(argument) : "unfolded");
    }
    assertEquals(List.of(0.75, 0.1 + 0.2, 0.4 + 0.5, 2, "unfolded", "unfolded"), arguments);
  }
}
