package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestTry extends TestJythonCallGraphShape {

  protected static final List<GraphAssertion> assertionsTry =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script try.py"}),
          new GraphAssertion(
              "script try.py",
              new String[] {"script try.py/test1", "script try.py/test2", "script try.py/test3"}),
          new GraphAssertion(
              "script try.py/test1",
              new String[] {
                "script try.py/test1/f1", "script try.py/test1/f2", "script try.py/test1/f3"
              }),
          new GraphAssertion(
              "script try.py/test2",
              new String[] {
                "script try.py/test2/f1", "script try.py/test2/f2", "script try.py/test2/f3"
              }),
          new GraphAssertion(
              "script try.py/test3",
              new String[] {
                "script try.py/test3/f1", "script try.py/test3/f2", "script try.py/test3/f3"
              }));

  @Test
  public void testTry()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("try.py");
    System.err.println(CG);
    verifyGraphAssertions(CG, assertionsTry);
  }
}
