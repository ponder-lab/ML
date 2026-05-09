package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestAssign extends TestJythonCallGraphShape {

  protected static final List<GraphAssertion> assertionsAssign1 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script assign1.py"}),
          new GraphAssertion(
              "script assign1.py", new String[] {"script assign1.py/f", "script assign1.py/g"}),
          new GraphAssertion("script assign1.py/f", new String[] {"script assign1.py/a"}),
          new GraphAssertion("script assign1.py/g", new String[] {"script assign1.py/a"}));

  @Test
  public void testAssign1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("assign1.py");
    verifyGraphAssertions(CG, assertionsAssign1);
  }

  protected static final List<GraphAssertion> assertionsAssign2 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script assign2.py"}),
          new GraphAssertion(
              "script assign2.py",
              new String[] {
                "script assign2.py/f", "script assign2.py/ff", "script assign2.py/fff"
              }),
          new GraphAssertion("script assign2.py/ff", new String[] {"script assign2.py/f"}),
          new GraphAssertion("script assign2.py/fff", new String[] {"script assign2.py/ff"}));

  @Test
  public void testAssign2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> E = makeEngine("assign2.py");
    PythonSSAPropagationCallGraphBuilder B = E.defaultCallGraphBuilder();
    CallGraph CG = B.makeCallGraph(B.getOptions());
    verifyGraphAssertions(CG, assertionsAssign2);
  }
}
