package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;

/**
 * Verification of <a href="https://github.com/wala/ML/issues/107">wala/ML#107</a> (method inherited
 * from parent class is missing from the call graph) and <a
 * href="https://github.com/wala/ML/issues/118">wala/ML#118</a> ({@code IClass.getSuperclass()}
 * returns {@code Object} instead of the declared Python parent) against current master. Both issues
 * are filed against an older state of the loader; this test pins their current status.
 */
public class TestIssue107And118 extends TestJythonCallGraphShape {

  private static final String FIXTURE = "test_issue107_118_repro.py";

  @Test
  public void issue107InheritedMethodAppearsInCallGraph()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph cg = this.process(FIXTURE);

    // Look for a CGNode whose method is `D.func` (the inherited method). If wala/ML#107
    // still reproduces, no such node exists.
    for (CGNode n : cg) {
      String name = n.getMethod().toString();
      if (name.contains("/D/func") || name.contains("/C/func")) return;
    }
    fail(
        "wala/ML#107: expected `D.func` (inherited via `class C(D)`) to appear in the call graph;"
            + " saw nodes: "
            + dumpMethodNames(cg));
  }

  @Test
  public void issue118SubclassSuperclassIsParent()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    IClassHierarchy cha = makeEngine(FIXTURE).defaultCallGraphBuilder().getClassHierarchy();

    IClass c =
        cha.lookupClass(
            TypeReference.findOrCreate(
                PythonTypes.pythonLoader, TypeName.string2TypeName("Lscript " + FIXTURE + "/C")));
    assertNotNull("expected to resolve `C` in the loaded class hierarchy", c);

    IClass parent = c.getSuperclass();
    assertNotNull("expected `C` to have a superclass", parent);
    String parentName = parent.getName().toString();
    assertEquals(
        "wala/ML#118: `C` should report `D` as its declared superclass, not `Object`",
        "Lscript " + FIXTURE + "/D",
        parentName);
  }

  private static String dumpMethodNames(CallGraph cg) {
    StringBuilder sb = new StringBuilder();
    for (CGNode n : cg) {
      IMethod m = n.getMethod();
      sb.append("\n  ").append(m.toString());
    }
    return sb.toString();
  }
}
