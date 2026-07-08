package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertFalse;

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
 * {@code gen1.py} and {@code gen3.py} exercise generator <em>expressions</em>, whose yielded
 * functions are not reachable on this front-end, a distinct modeling gap pinned by the tests below.
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
   * called. Unlike the generator function of {@code gen2.py}, the expression's yielded functions
   * {@code f1}/{@code f2}/{@code f3} are <em>not</em> reachable; only the script itself is. This
   * pins the current behavior. TODO(<a
   * href="https://github.com/wala/ML/issues/701">wala/ML#701</a>): extend to reach {@code
   * f1}/{@code f2}/{@code f3} once generator expressions are modeled at the call-graph level.
   */
  protected static final List<GraphAssertion> assertionsGen1 =
      List.of(new GraphAssertion(ROOT, new String[] {"script gen1.py"}));

  /**
   * {@code gen3.py}: a generator <em>expression</em> returned from {@code makeGenerator(fs)}.
   * {@code makeGenerator} is reachable, but the expression's yielded functions {@code f1}/{@code
   * f2}/{@code f3} are not reachable, the same generator-expression gap as {@code gen1.py}. TODO(<a
   * href="https://github.com/wala/ML/issues/701">wala/ML#701</a>): extend to reach {@code
   * f1}/{@code f2}/{@code f3} once generator expressions are modeled at the call-graph level.
   */
  protected static final List<GraphAssertion> assertionsGen3 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script gen3.py"}),
          new GraphAssertion("script gen3.py", new String[] {"script gen3.py/makeGenerator"}));

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
    // The generator expression's yielded functions and their returned lambdas are not reachable;
    // assert their absence so the gap cannot silently regress. When wala/ML#701 is fixed these will
    // start failing, cueing an update to positive reachability.
    assertNodesAbsent(
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
    // As with gen1.py, the generator expression's yielded functions and their returned lambdas are
    // not reachable; assert their absence so the gap cannot silently regress (wala/ML#701).
    assertNodesAbsent(
        cg,
        "Lscript gen3.py/f1",
        "Lscript gen3.py/f2",
        "Lscript gen3.py/f3",
        "Lscript gen3.py/f1/lambda1",
        "Lscript gen3.py/f2/lambda1",
        "Lscript gen3.py/f3/lambda1");
  }

  /**
   * Asserts that no call-graph node is declared by any of the given classes.
   *
   * @param cg The call graph to check.
   * @param absentClassNames The qualified declaring-class names expected to have no node.
   */
  private static void assertNodesAbsent(CallGraph cg, String... absentClassNames) {
    Set<String> declaringClasses = new HashSet<>();
    for (CGNode node : cg)
      declaringClasses.add(node.getMethod().getDeclaringClass().getName().toString());
    for (String absent : absentClassNames)
      assertFalse(
          "Expected no call-graph node declared by " + absent, declaringClasses.contains(absent));
  }
}
