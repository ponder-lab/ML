package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that an {@code iter} or {@code next} call records whether the front end made it while
 * lowering a {@code for} loop or a comprehension's generator to the iteration protocol, or whether
 * it is a call in the source (<a href="https://github.com/wala/ML/issues/1131">wala/ML#1131</a>).
 * The two lower to the same read of the builtin's name and the same invoke.
 */
public class TestIterationProtocolCalls extends AbstractTensorTest {

  private static final String FILE = "tf2_test_iteration_protocol_calls.py";

  /** An explicit {@code next(iter(x))}: both calls are the source's. */
  @Test
  public void testExplicitCalls() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of("iter:source", "next:source"), origins("iter_param"));
  }

  /** A {@code for} loop over a parameter: both calls are the lowering's. */
  @Test
  public void testForLoop() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of("iter:lowered", "next:lowered"), origins("iterate_param"));
  }

  /**
   * A list comprehension makes no {@code iter} or {@code next} call: its generator lowers to a
   * comprehension function that the call graph builder iterates, so no call needs telling apart.
   */
  @Test
  public void testListComprehension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), origins("list_comp"));
  }

  /** A set comprehension makes no {@code iter} or {@code next} call, as a list comprehension. */
  @Test
  public void testSetComprehension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), origins("set_comp"));
  }

  /** A dict comprehension's generator: both calls are the lowering's. */
  @Test
  public void testDictComprehension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of("iter:lowered", "next:lowered"), origins("dict_comp"));
  }

  /**
   * A generator expression makes no {@code iter} or {@code next} call: it lowers like a list
   * comprehension (wala/ML#701).
   */
  @Test
  public void testGeneratorExpression()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), origins("gen_exp"));
  }

  /**
   * The origin of each {@code iter} or {@code next} call in a function of the fixture, including
   * the functions the front end makes for its comprehensions.
   *
   * @param function The function's name in the fixture.
   * @return Each call's builtin and origin, as {@code name:lowered} or {@code name:source}.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private Set<String> origins(String function)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String prefix = "script " + FILE + "." + function + ".";
    Set<String> ret = new HashSet<>();
    List<String> seen = new ArrayList<>();
    for (CGNode node : cg) {
      String signature = node.getMethod().getSignature();
      if (!signature.startsWith(prefix) && !signature.startsWith("script " + FILE + "/" + function))
        continue;
      seen.add(signature);
      IR ir = node.getIR();
      if (ir == null) continue;
      DefUse du = new DefUse(ir);
      for (SSAInstruction instruction : ir.getInstructions())
        if (instruction instanceof PythonInvokeInstruction call
            && du.getDef(call.getUse(0)) instanceof AstLexicalRead read)
          for (int i = 0; i < read.getAccessCount(); i++) {
            String name = read.getAccess(i).variableName();
            if (name.equals("iter") || name.equals("next"))
              ret.add(name + (call.isIterationProtocol() ? ":lowered" : ":source"));
          }
    }
    assertTrue("`" + function + "` must be in the call graph.", !seen.isEmpty());
    return ret;
  }
}
