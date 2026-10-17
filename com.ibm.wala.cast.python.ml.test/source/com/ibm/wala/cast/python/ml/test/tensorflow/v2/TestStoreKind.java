package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
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
 * Tests that a property read or write records whether it came from subscript syntax, {@code y[k]},
 * or attribute syntax, {@code y.k} (<a
 * href="https://github.com/wala/ML/issues/1100">wala/ML#1100</a>). The two lower to the same
 * instruction, and with a string-constant key, as in {@code y["tag"]} and {@code y.tag}, they are
 * otherwise identical.
 */
public class TestStoreKind extends AbstractTensorTest {

  private static final String FILE = "tf2_test_store_kind.py";

  /**
   * Each write in {@code stores}, by its member, in instruction order: the member's string
   * constant, or {@code "i"} for the parameter key, with whether it is a subscript.
   */
  @Test
  public void testWrites() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of("tag:attribute", "tag:subscript", "i:subscript", "n:subscript", "n:subscript"),
        kinds(true));
  }

  /** As {@link #testWrites()}, for the reads, including the augmented store's read of {@code n}. */
  @Test
  public void testReads() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of("n:subscript", "tag:attribute", "tag:subscript", "i:subscript"), kinds(false));
  }

  /**
   * The member and kind of each property write or read in {@code stores}'s IR, in instruction
   * order.
   *
   * @param writes Whether to collect the writes rather than the reads.
   * @return Each instruction's member and kind, as {@code member:subscript} or {@code
   *     member:attribute}.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private List<String> kinds(boolean writes)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String signature = "script " + FILE + ".stores.do()LRoot;";
    IR ir = null;
    for (CGNode node : cg)
      if (node.getMethod().getSignature().equals(signature)) {
        ir = node.getIR();
        break;
      }
    assertTrue("`stores` must be in the call graph.", ir != null);
    SymbolTable symbols = ir.getSymbolTable();
    // The key parameter, the second parameter after the function object.
    int key = symbols.getParameter(2);
    List<String> ret = new ArrayList<>();
    for (SSAInstruction instruction : ir.getInstructions()) {
      int member;
      boolean subscript;
      if (writes && instruction instanceof PythonPropertyWrite write) {
        member = write.getMemberRef();
        subscript = write.isSubscript();
      } else if (!writes && instruction instanceof PythonPropertyRead read) {
        member = read.getMemberRef();
        subscript = read.isSubscript();
      } else continue;
      String name =
          member == key
              ? "i"
              : symbols.isStringConstant(member) ? symbols.getStringValue(member) : null;
      if (name == null) continue;
      ret.add(name + (subscript ? ":subscript" : ":attribute"));
    }
    return ret;
  }
}
