package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * Tests that a property read records whether it came from destructuring, {@code a, b = y}, rather
 * than from attribute syntax, {@code y.k}, or subscript syntax, {@code y[k]} (<a
 * href="https://github.com/wala/ML/issues/1117">wala/ML#1117</a>). All three lower to the same
 * instruction.
 */
public class TestReadKind extends AbstractTensorTest {

  private static final String FILE = "tf2_test_read_kind.py";

  /**
   * Each read in {@code reads}, in instruction order, by its kind and member: a destructuring of a
   * tuple, of a list target, of a {@code for} target and of a nested target, then a subscript by a
   * constant and by the parameter {@code k}, and an attribute read.
   */
  @Test
  public void testReads() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of(
            "unpacking:0",
            "unpacking:1",
            "unpacking:0",
            "unpacking:1",
            "subscript:0",
            "subscript:k",
            "attribute:shape",
            "unpacking:0",
            "unpacking:1",
            "unpacking:0",
            "unpacking:1",
            "unpacking:0",
            "unpacking:1"),
        kinds());
  }

  /**
   * The kind and member of each property read in {@code reads}'s IR, in instruction order.
   *
   * @return Each read as {@code kind:member}, the kind {@code unpacking}, {@code subscript} or
   *     {@code attribute}, and the member a constant's value or {@code k} for that parameter.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private List<String> kinds() throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String signature = "script " + FILE + ".reads.do()LRoot;";
    IR ir = null;
    for (CGNode node : cg)
      if (node.getMethod().getSignature().equals(signature)) {
        ir = node.getIR();
        break;
      }
    assertTrue("`reads` must be in the call graph.", ir != null);
    SymbolTable symbols = ir.getSymbolTable();
    // The key parameter, the fifth parameter after the function object.
    int key = symbols.getParameter(5);
    List<String> ret = new ArrayList<>();
    for (SSAInstruction instruction : ir.getInstructions()) {
      if (!(instruction instanceof PythonPropertyRead read)) continue;
      int member = read.getMemberRef();
      String name =
          member == key
              ? "k"
              : symbols.isConstant(member)
                  ? String.valueOf(symbols.getConstantValue(member))
                  : null;
      if (name == null) continue;
      String kind =
          read.isUnpacking() ? "unpacking" : read.isSubscript() ? "subscript" : "attribute";
      ret.add(kind + ":" + name);
    }
    return ret;
  }
}
