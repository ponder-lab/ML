package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestFor extends TestJythonCallGraphShape {

  protected static final List<GraphAssertion> assertionsFor1 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script for1.py"}),
          new GraphAssertion(
              "script for1.py",
              new String[] {"script for1.py/f1", "script for1.py/f2", "script for1.py/f3"}));

  @Test
  public void testFor1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("for1.py");
    verifyGraphAssertions(CG, assertionsFor1);
  }

  protected static final List<GraphAssertion> assertionsFor2 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script for2.py"}),
          new GraphAssertion("script for2.py", new String[] {"script for2.py/doit"}),
          new GraphAssertion(
              "script for2.py/doit",
              new String[] {"script for2.py/f1", "script for2.py/f2", "script for2.py/f3"}));

  @Test
  public void testFor2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("for2.py");
    verifyGraphAssertions(CG, assertionsFor2);
  }

  protected static final List<GraphAssertion> assertionsFor3 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script for3.py"}),
          new GraphAssertion(
              "script for3.py",
              new String[] {"script for3.py/g1", "script for3.py/g2", "script for3.py/g3"}),
          new GraphAssertion(
              "script for3.py/g1",
              new String[] {"script for3.py/f1", "script for3.py/f2", "script for3.py/f3"}),
          new GraphAssertion(
              "script for3.py/g2",
              new String[] {"script for3.py/f1", "script for3.py/f2", "script for3.py/f3"}),
          new GraphAssertion(
              "script for3.py/g3",
              new String[] {"script for3.py/f1", "script for3.py/f2", "script for3.py/f3"}));

  @Test
  public void testFor3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("for3.py");
    verifyGraphAssertions(CG, assertionsFor3);
  }

  protected static final List<GraphAssertion> assertionsFor4 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script for4.py"}),
          new GraphAssertion("script for4.py", new String[] {"script for4.py/lambda1"}));

  protected static final List<GraphAssertion> assertionsFor6 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script for6.py"}),
          new GraphAssertion(
              "script for6.py", new String[] {"script for6.py/mf/lambda1", "script for6.py/mf"}));

  @Test
  public void testFor6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("for6.py");
    verifyGraphAssertions(CG, assertionsFor6);
  }
}
