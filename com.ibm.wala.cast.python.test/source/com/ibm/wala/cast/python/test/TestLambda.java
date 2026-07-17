package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestLambda extends TestJythonCallGraphShape {

  protected static final List<GraphAssertion> assertionsLambda1 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script lambda1.py"}),
          new GraphAssertion(
              "script lambda1.py",
              new String[] {
                "script lambda1.py/Foo",
                "$script lambda1.py/Foo/foo:trampoline2$b",
                "script lambda1.py/lambda1",
                "script lambda1.py/lambda2"
              }),
          new GraphAssertion(
              "$script lambda1.py/Foo/foo:trampoline2$b",
              new String[] {"script lambda1.py/Foo/foo"}));

  @Test
  public void testLambda1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("lambda1.py");
    verifyGraphAssertions(CG, assertionsLambda1);
  }

  protected static final List<GraphAssertion> assertionsLambda2 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script lambda2.py"}),
          new GraphAssertion(
              "script lambda2.py",
              new String[] {
                "script lambda2.py/Foo",
                "$script lambda2.py/Foo/foo:trampoline2$b",
                "script lambda2.py/lambda2",
                "script lambda2.py/lambda3"
              }),
          new GraphAssertion(
              "$script lambda2.py/Foo/foo:trampoline2$b",
              new String[] {"script lambda2.py/Foo/foo"}),
          new GraphAssertion(
              "script lambda2.py/lambda2", new String[] {"script lambda2.py/lambda1"}),
          new GraphAssertion(
              "script lambda2.py/lambda3", new String[] {"script lambda2.py/lambda1"}));

  @Test
  public void testLambda2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("lambda2.py");
    verifyGraphAssertions(CG, assertionsLambda2);
  }

  protected static final List<GraphAssertion> assertionsLambda3 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script lambda3.py"}),
          new GraphAssertion(
              "script lambda3.py",
              new String[] {"script lambda3.py/Foo", "$script lambda3.py/Foo/foo:trampoline1$a$b"}),
          new GraphAssertion(
              "$script lambda3.py/Foo/foo:trampoline1$a$b",
              new String[] {"script lambda3.py/Foo/foo"}),
          new GraphAssertion(
              "script lambda3.py/Foo/foo", new String[] {"script lambda3.py/lambda3"}),
          new GraphAssertion(
              "script lambda3.py/lambda3",
              new String[] {"script lambda3.py/lambda1", "script lambda3.py/lambda2"}));

  @Test
  public void testLambda3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = this.makeEngine("lambda3.py");
    PropagationCallGraphBuilder callGraphBuilder = engine.defaultCallGraphBuilder();
    CallGraph CG = callGraphBuilder.makeCallGraph(callGraphBuilder.getOptions());

    /*
       CAstCallGraphUtil.AVOID_DUMP.set(false);
        CAstCallGraphUtil.dumpCG(
            (SSAContextInterpreter) callGraphBuilder.getContextInterpreter(),
            callGraphBuilder.getPointerAnalysis(),
            CG);
    */

    verifyGraphAssertions(CG, assertionsLambda3);
  }

  protected static final List<GraphAssertion> assertionsLambda4 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script lambda4.py"}),
          new GraphAssertion("script lambda4.py", new String[] {"script lambda4.py/lambda2"}),
          new GraphAssertion(
              "script lambda4.py/lambda2",
              new String[] {"script lambda4.py/lambda1", "script lambda4.py/lambda3"}));

  @Test
  public void testLambda4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = this.makeEngine("lambda4.py");
    PropagationCallGraphBuilder callGraphBuilder = engine.defaultCallGraphBuilder();
    CallGraph CG = callGraphBuilder.makeCallGraph(callGraphBuilder.getOptions());

    /*
       CAstCallGraphUtil.AVOID_DUMP.set(false);
        CAstCal8lGraphUtil.dumpCG(
            (SSAContextInterpreter) callGraphBuilder.getContextInterpreter(),
            callGraphBuilder.getPointerAnalysis(),
            CG);
    */

    verifyGraphAssertions(CG, assertionsLambda4);
  }
}
