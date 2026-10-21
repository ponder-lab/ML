package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ssa.PythonInvokeInstruction.IterationSource.COMPREHENSION;
import static com.ibm.wala.cast.python.ssa.PythonInvokeInstruction.IterationSource.FOR_STATEMENT;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ssa.IterationSites;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;

/**
 * Tests that each iteration site records whether the source wrote a {@code for} statement or a
 * comprehension, and which values it iterates (<a
 * href="https://github.com/wala/ML/issues/1133">wala/ML#1133</a>).
 */
public class TestIterationSites extends AbstractTensorTest {

  private static final String FILE = "tf2_test_iteration_sites.py";

  /** Both element gets of a {@code for} statement step a loop over the parameter. */
  @Test
  public void testForStatement() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(List.of("FOR_STATEMENT over x, 2 gets"), loops("for_statement"));
  }

  /** A dict comprehension lowers to a loop as a {@code for} statement does, but records itself. */
  @Test
  public void testDictComprehension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(List.of("COMPREHENSION over x, 2 gets"), loops("dict_comp"));
  }

  /**
   * A {@code for} statement and a dict comprehension over one variable stay two loops, each with
   * its own source, though they iterate the same value.
   */
  @Test
  public void testSameVariable() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of("COMPREHENSION over x, 2 gets", "FOR_STATEMENT over x, 2 gets"),
        loops("same_variable"));
  }

  /** A nested {@code for} statement's element gets step the inner loop, not the outer one. */
  @Test
  public void testNested() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of("FOR_STATEMENT over x, 2 gets", "FOR_STATEMENT over y, 2 gets"), loops("nested"));
  }

  /**
   * A {@code for} statement over a dict comprehension: the comprehension's loop runs while the
   * statement's iterable is evaluated, before the statement's own {@code iter} call.
   */
  @Test
  public void testForOverDictComprehension()
      throws ClassHierarchyException, CancelException, IOException {
    List<String> loops = loops("for_over_dict_comp");
    assertEquals(2, loops.size());
    assertEquals("COMPREHENSION over x, 2 gets", loops.get(0));
    assertTrue(loops.get(1), loops.get(1).startsWith("FOR_STATEMENT over "));
  }

  /**
   * A list comprehension with two generators lowers to a single invoke, which records itself and
   * its iterables in source order.
   */
  @Test
  public void testListComprehension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(List.of("x", "y"), comprehensionIterables("list_comp"));
  }

  /** A set comprehension lowers to a single invoke, as a list comprehension does. */
  @Test
  public void testSetComprehension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(List.of("x"), comprehensionIterables("set_comp"));
  }

  /** A generator expression lowers to a single invoke, as a list comprehension does. */
  @Test
  public void testGeneratorExpression()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(List.of("x"), comprehensionIterables("gen_exp"));
  }

  /** An {@code iter} or {@code next} call in the source records no iteration source. */
  @Test
  public void testExplicitIter() throws ClassHierarchyException, CancelException, IOException {
    IR ir = ir("explicit_iter");
    int calls = 0;
    for (SSAInstruction instruction : ir.getInstructions())
      if (instruction instanceof PythonInvokeInstruction call) {
        calls++;
        assertNull(call.getIterationSource());
        assertFalse(call.isComprehension());
        assertFalse(call.isIterationProtocol());
      }
    assertTrue("`explicit_iter` must make calls.", calls > 0);
    assertEquals(List.of(), loops("explicit_iter"));
  }

  /**
   * Describes the loops of a fixture function, each with its source, its iterable and the number of
   * its element gets, in sorted order.
   *
   * @param function The function's name in the fixture.
   * @return The loops' descriptions.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private List<String> loops(String function)
      throws ClassHierarchyException, CancelException, IOException {
    IR ir = ir(function);
    IterationSites sites = new IterationSites(ir);
    Map<Integer, List<EachElementGetInstruction>> byCall = new TreeMap<>();
    for (SSAInstruction instruction : ir.getInstructions())
      if (instruction instanceof EachElementGetInstruction get) {
        PythonInvokeInstruction call = sites.getIteratorCall(get);
        assertNotNull("Every element get of `" + function + "` steps a loop: " + get, call);
        assertTrue(call.isIterationProtocol());
        assertEquals(call.getIterationSource(), sites.getIterationSource(get));
        byCall.computeIfAbsent(call.iIndex(), i -> new ArrayList<>()).add(get);
      }
    List<String> ret = new ArrayList<>();
    for (List<EachElementGetInstruction> gets : byCall.values()) {
      EachElementGetInstruction get = gets.get(0);
      ret.add(
          sites.getIterationSource(get)
              + " over "
              + name(ir, sites.getIterable(get))
              + ", "
              + gets.size()
              + " gets");
    }
    Collections.sort(ret);
    return ret;
  }

  /**
   * The iterables of the one comprehension invoke in a fixture function, by parameter name.
   *
   * @param function The function's name in the fixture.
   * @return The iterables' names, in source order.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private List<String> comprehensionIterables(String function)
      throws ClassHierarchyException, CancelException, IOException {
    IR ir = ir(function);
    List<PythonInvokeInstruction> comprehensions = new ArrayList<>();
    for (SSAInstruction instruction : ir.getInstructions()) {
      if (instruction instanceof PythonInvokeInstruction call && call.isComprehension()) {
        assertEquals(COMPREHENSION, call.getIterationSource());
        assertFalse(call.isIterationProtocol());
        comprehensions.add(call);
      }
      if (instruction instanceof PythonInvokeInstruction call && call.isIterationProtocol())
        assertEquals(FOR_STATEMENT, call.getIterationSource());
      assertFalse(
          "A list, set or generator comprehension lowers to no loop in `" + function + "`.",
          instruction instanceof EachElementGetInstruction);
    }
    assertEquals(1, comprehensions.size());
    List<String> ret = new ArrayList<>();
    for (int iterable : comprehensions.get(0).getComprehensionIterables())
      ret.add(name(ir, iterable));
    return ret;
  }

  /**
   * The parameter name of a value number in a fixture function, or the value number itself when it
   * is not a parameter.
   *
   * @param ir The function's IR.
   * @param valueNumber The value number.
   * @return The name.
   */
  private static String name(IR ir, int valueNumber) {
    String[] names = {"x", "y"};
    // A Python function's first parameter is the function object itself.
    for (int i = 1; i < ir.getNumberOfParameters(); i++)
      if (ir.getParameter(i) == valueNumber) return names[i - 1];
    return "v" + valueNumber;
  }

  /**
   * The IR of a fixture function.
   *
   * @param function The function's name in the fixture.
   * @return Its IR.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private IR ir(String function) throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String prefix = "script " + FILE + "." + function + ".";
    for (CGNode node : cg)
      if (node.getMethod().getSignature().startsWith(prefix)) {
        assertNotNull(node.getIR());
        return node.getIR();
      }
    throw new AssertionError("`" + function + "` must be in the call graph.");
  }
}
