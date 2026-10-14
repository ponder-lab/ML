package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;

/**
 * Tests that a function writes, to its lexical scope, a name an enclosed function reads from it: an
 * enclosed comprehension's element or filter function as a lambda or a nested function does (<a
 * href="https://github.com/wala/ML/issues/1066">wala/ML#1066</a>).
 */
public class TestComprehensionCapture extends TestJythonCallGraphShape {

  private static final String FILE = "comp_capture.py";

  /**
   * Whether the given function's IR writes its parameter {@code x} to its lexical scope.
   *
   * @param function The function's name in the fixture.
   * @return {@code true} iff some node of the function writes {@code x} lexically.
   */
  private boolean writesX(String function)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = this.makeEngine(FILE);
    PropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    boolean found = false;
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + function))
        continue;
      reached = true;
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions())
        if (instruction instanceof AstLexicalWrite write)
          for (Access access : write.getAccesses())
            if (access.variableName().equals("x")) found = true;
    }
    assertTrue(function + " is not in the call graph", reached);
    return found;
  }

  /** A lambda reading {@code x}: the enclosing function writes it. */
  @Test
  public void testLambda()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertTrue(this.writesX("via_lambda"));
  }

  /** A nested function reading {@code x}: the enclosing function writes it. */
  @Test
  public void testNestedFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertTrue(this.writesX("via_nested_def"));
  }

  /** A comprehension's element reading {@code x}: the enclosing function writes it. */
  @Test
  public void testComprehensionElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertTrue(this.writesX("via_comprehension"));
  }

  /** A comprehension's filter reading {@code x}: the enclosing function writes it. */
  @Test
  public void testComprehensionFilter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertTrue(this.writesX("via_filter"));
  }
}
