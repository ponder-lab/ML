package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * Call-graph reachability guards for the three generator fixtures inherited from upstream. {@code
 * gen2.py} (a generator <em>function</em> yielding functions) is the call-graph shape that <a
 * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>'s modeling supports: iterating
 * {@code gen()} reaches the yielded {@code f1}/{@code f2}/{@code f3} and their returned lambdas.
 * {@code gen1.py} and {@code gen3.py} exercise generator <em>expressions</em>, which lower through
 * the comprehension machinery (wala/ML#701) and reach their yielded functions the same way.
 */
public class TestGenerators extends TestJythonCallGraphShape {

  /**
   * {@code gen2.py}: {@code for f in gen(): g = f(i); g(i)} over a generator function that yields
   * {@code f1}/{@code f2}/{@code f3}. The yielded functions and their returned lambdas are all
   * reachable, giving a call-graph-level regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a> (the existing guards assert tensor
   * typing, not call-graph reachability).
   */
  protected static final List<GraphAssertion> assertionsGen2 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script gen2.py"}),
          new GraphAssertion(
              "script gen2.py",
              new String[] {
                "script gen2.py/gen",
                "script gen2.py/f1",
                "script gen2.py/f2",
                "script gen2.py/f3",
                "script gen2.py/f1/lambda1",
                "script gen2.py/f2/lambda1",
                "script gen2.py/f3/lambda1"
              }));

  /**
   * {@code gen1.py}: a generator <em>expression</em> {@code (f(3) for f in fs)} iterated and
   * called. The expression lowers through the comprehension machinery (wala/ML#701), so the
   * comprehension body reaches the yielded {@code f1}/{@code f2}/{@code f3} and the iterating
   * script reaches their returned lambdas, matching the generator-function shape of {@code
   * gen2.py}.
   */
  protected static final List<GraphAssertion> assertionsGen1 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script gen1.py"}),
          // The script reaches the comprehension body through the synthetic trampoline (see
          // PythonComprehensionTrampolines), so no direct script-to-comprehension edge is
          // asserted; the body's outgoing edges and the node-presence checks below pin it.
          new GraphAssertion(
              "script gen1.py",
              new String[] {
                "script gen1.py/f1/lambda1",
                "script gen1.py/f2/lambda1",
                "script gen1.py/f3/lambda1"
              }),
          new GraphAssertion(
              "script gen1.py/comprehension1",
              new String[] {"script gen1.py/f1", "script gen1.py/f2", "script gen1.py/f3"}));

  /**
   * {@code gen3.py}: a generator <em>expression</em> returned from {@code makeGenerator(fs)}. The
   * expression lowers through the comprehension machinery (wala/ML#701), so its body (declared
   * under {@code makeGenerator}) reaches the yielded {@code f1}/{@code f2}/{@code f3} and the
   * iterating script reaches their returned lambdas, the same shape as {@code gen1.py} across the
   * function boundary.
   */
  protected static final List<GraphAssertion> assertionsGen3 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script gen3.py"}),
          new GraphAssertion("script gen3.py", new String[] {"script gen3.py/makeGenerator"}),
          new GraphAssertion(
              "script gen3.py/makeGenerator/comprehension1",
              new String[] {"script gen3.py/f1", "script gen3.py/f2", "script gen3.py/f3"}));

  /**
   * Guards {@code gen2.py}'s call-graph shape; see {@link #assertionsGen2}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGen2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    verifyGraphAssertions(process("gen2.py"), assertionsGen2);
  }

  /**
   * Pins {@code gen1.py}'s call-graph shape; see {@link #assertionsGen1}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGen1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph cg = process("gen1.py");
    verifyGraphAssertions(cg, assertionsGen1);
    // The generator expression's yielded functions and their returned lambdas are all reachable
    // (wala/ML#701); assert their presence so the fix cannot silently regress.
    assertNodesPresent(
        cg,
        "Lscript gen1.py/f1",
        "Lscript gen1.py/f2",
        "Lscript gen1.py/f3",
        "Lscript gen1.py/f1/lambda1",
        "Lscript gen1.py/f2/lambda1",
        "Lscript gen1.py/f3/lambda1");
  }

  /**
   * Pins {@code gen3.py}'s call-graph shape; see {@link #assertionsGen3}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGen3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph cg = process("gen3.py");
    verifyGraphAssertions(cg, assertionsGen3);
    // As with gen1.py, the generator expression's yielded functions and their returned lambdas
    // are all reachable (wala/ML#701); assert their presence so the fix cannot silently regress.
    assertNodesPresent(
        cg,
        "Lscript gen3.py/f1",
        "Lscript gen3.py/f2",
        "Lscript gen3.py/f3",
        "Lscript gen3.py/f1/lambda1",
        "Lscript gen3.py/f2/lambda1",
        "Lscript gen3.py/f3/lambda1");
  }

  /**
   * Asserts that some call-graph node is declared by each of the given classes.
   *
   * @param cg The call graph to check.
   * @param presentClassNames The qualified declaring-class names expected to have a node.
   */
  private static void assertNodesPresent(CallGraph cg, String... presentClassNames) {
    Set<String> declaringClasses = new HashSet<>();
    for (CGNode node : cg) {
      declaringClasses.add(node.getMethod().getDeclaringClass().getName().toString());
    }
    for (String present : presentClassNames) {
      assertTrue(
          "Expected a call-graph node declared by " + present, declaringClasses.contains(present));
    }
  }
}
