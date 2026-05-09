package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestLibrary extends TestPythonLibraryCallGraphShape {

  protected static final List<GraphAssertion> assertionsLib1 =
      List.of(
          new GraphAssertion(
              ROOT,
              new String[] {
                "script lib1.py", "script lib1.py/es1", "script lib1.py/es2", "script lib1.py/es3"
              }),
          new GraphAssertion("script lib1.py/es1", new String[] {"turtle:turtle"}),
          new GraphAssertion("script lib1.py/es2", new String[] {"turtle:turtle"}),
          new GraphAssertion("script lib1.py/es3", new String[] {"turtle:turtle"}));

  @Test
  public void testLib1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("lib1.py");
    System.err.println(CG);
    verifyGraphAssertions(CG, assertionsLib1);
  }

  protected static final List<GraphAssertion> assertionsLib2 =
      List.of(
          new GraphAssertion(
              ROOT,
              new String[] {
                "script lib2.py",
                "script lib2.py/Lib",
                "$script lib2.py/Lib/es1:trampoline4",
                "$script lib2.py/Lib/es2:trampoline3",
                "$script lib2.py/Lib/es3:trampoline3"
              }),
          new GraphAssertion(
              "$script lib2.py/Lib/es1:trampoline4", new String[] {"script lib2.py/Lib/es1"}),
          new GraphAssertion(
              "$script lib2.py/Lib/es2:trampoline3", new String[] {"script lib2.py/Lib/es2"}),
          new GraphAssertion(
              "$script lib2.py/Lib/es3:trampoline3", new String[] {"script lib2.py/Lib/es3"}),
          new GraphAssertion("script lib2.py/Lib/es1", new String[] {"turtle:turtle"}),
          new GraphAssertion("script lib2.py/Lib/es2", new String[] {"turtle:turtle"}),
          new GraphAssertion("script lib2.py/Lib/es3", new String[] {"turtle:turtle"}));

  @Test
  public void testLib2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("lib2.py");
    System.err.println(CG);
    CG.forEach(
        (n) -> {
          System.err.println(n.getIR());
        });
    verifyGraphAssertions(CG, assertionsLib2);
  }
}
