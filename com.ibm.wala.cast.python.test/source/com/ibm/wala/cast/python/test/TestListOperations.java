package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * List repetition and concatenation carry their operands' elements (wala/ML#960). The fixture
 * reaches {@code Block.call} only through {@code [None] * n} and {@code sink} only through {@code
 * [1] + [2]}, so each parameter's points-to set is evidence of the model and empty without it.
 */
public class TestListOperations extends TestJythonCallGraphShape {

  /**
   * A parameter bound from an iterated repetition of {@code [None]} points to the None constant,
   * and to nothing else.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepetitionCarriesNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<Object> values = lastParameterValues("list_operations_pa.py", "/Block/call");
    assertFalse("the repeated None never reached `past`", values.isEmpty());
    assertTrue(
        "`past` reads " + values + ", not the None constant alone", values.equals(nullOnly()));
  }

  /**
   * A parameter bound from an iterated concatenation of {@code [1]} and {@code [2]} points to both
   * integer constants.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testConcatenationCarriesBothElements()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<Object> values = lastParameterValues("list_operations_pa.py", "/sink");
    assertTrue("`v` reads " + values, values.contains(1L) || values.contains(1));
    assertTrue("`v` reads " + values, values.contains(2L) || values.contains(2));
  }

  /**
   * A repetition whose list operand arrives as a parameter, a represented key rather than a
   * literal, still carries the element: the model learns the operand's keys by a side effect.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepetitionOfParameterCarriesElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<Object> values = lastParameterValues("list_operations_pa.py", "/sink2");
    assertTrue("`v` reads " + values, values.contains(7L) || values.contains(7));
  }

  /**
   * A concatenation whose left operand is a literal and whose right operand's list arrives through
   * a call chain, after the literal was offered: the literal's element is retried when the right
   * operand's set grows, so the sink reads both elements.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testConcatenationWithLateRightOperandCarriesBothElements()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<Object> values = lastParameterValues("list_operations_pa.py", "/sink3");
    assertTrue("`v` reads " + values, values.contains(1L) || values.contains(1));
    assertTrue("`v` reads " + values, values.contains(2L) || values.contains(2));
  }

  private static Set<Object> nullOnly() {
    Set<Object> ret = new HashSet<>();
    ret.add(null);
    return ret;
  }

  /**
   * The constant values the last parameter of every node of the named function points to, over all
   * of the function's nodes; a non-constant member is recorded as its class.
   */
  private Set<Object> lastParameterValues(String file, String functionSuffix)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine(file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Set<Object> values = new HashSet<>();
    for (CGNode node : cg) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      if (!name.endsWith(functionSuffix)) continue;
      IR ir = node.getIR();
      if (ir == null || ir.getNumberOfParameters() == 0) continue;
      int vn = ir.getParameter(ir.getNumberOfParameters() - 1);
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(node, vn);
      for (InstanceKey ik : pa.getPointsToSet(pk))
        values.add(ik instanceof ConstantKey ? ((ConstantKey<?>) ik).getValue() : ik.getClass());
    }
    return values;
  }
}
