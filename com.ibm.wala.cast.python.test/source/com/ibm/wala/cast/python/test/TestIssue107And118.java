package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
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

  /**
   * Asserts the call-graph EDGES from {@code c.func(5)} reach the inherited body. The bug was a
   * missing edge, not just a missing node; node-existence alone could be satisfied by an unrelated
   * reachability path.
   */
  private static final List<GraphAssertion> assertions =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script " + FIXTURE}),
          // Script-level callees: the `c = C()` constructor call and the `c.func(5)`
          // trampoline. `D` is declared but never instantiated, so no edge from script to D.
          // The trampoline edge is the load-bearing assertion for wala/ML#107: pre-fix, this
          // edge was missing entirely (no `c.func(5)` resolution in the call graph at all).
          new GraphAssertion(
              "script " + FIXTURE,
              new String[] {
                "script " + FIXTURE + "/C", "$script " + FIXTURE + "/D/func:trampoline2"
              }),
          // The trampoline must in turn dispatch to `D.func`'s body. Pre-fix, even if the
          // trampoline node existed, this resolution edge would not.
          new GraphAssertion(
              "$script " + FIXTURE + "/D/func:trampoline2",
              new String[] {"script " + FIXTURE + "/D/func"}));

  @Test
  public void issue107InheritedMethodEdgeReachesParentBody()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph cg = this.process(FIXTURE);
    verifyGraphAssertions(cg, assertions);
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
}
