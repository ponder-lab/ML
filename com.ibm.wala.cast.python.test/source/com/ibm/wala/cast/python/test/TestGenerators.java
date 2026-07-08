package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

/**
 * Call-graph reachability guards for the three generator fixtures inherited from upstream. {@code
 * gen2.py} (a generator <em>function</em> yielding functions) is the call-graph shape that <a
 * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>'s modeling supports: iterating
 * {@code gen()} reaches the yielded {@code f1}/{@code f2}/{@code f3} and their returned lambdas.
 * {@code gen1.py} and {@code gen3.py} exercise generator <em>expressions</em>, whose yielded
 * functions are not reachable on this front-end — a distinct modeling gap pinned by the tests
 * below.
 */
public class TestGenerators extends TestJythonCallGraphShape {

  /**
   * {@code gen2.py}: {@code for f in gen(): g = f(i); g(i)} over a generator function that yields
   * {@code f1}/{@code f2}/{@code f3}. The yielded functions and their returned lambdas are all
   * reachable — a call-graph-level regression guard for <a
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
   * pins the current behavior. TODO: extend to reach {@code f1}/{@code f2}/{@code f3} once
   * generator expressions are modeled at the call-graph level.
   */
  protected static final List<GraphAssertion> assertionsGen1 =
      List.of(new GraphAssertion(ROOT, new String[] {"script gen1.py"}));

  /**
   * {@code gen3.py}: a generator <em>expression</em> returned from {@code makeGenerator()}. {@code
   * makeGenerator} is reachable, but the expression's yielded functions {@code f1}/{@code
   * f2}/{@code f3} are not — the same generator-expression gap as {@code gen1.py}. TODO: extend to
   * reach {@code f1}/{@code f2}/{@code f3} once generator expressions are modeled at the call-graph
   * level.
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
    verifyGraphAssertions(process("gen1.py"), assertionsGen1);
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
    verifyGraphAssertions(process("gen3.py"), assertionsGen3);
  }
}
