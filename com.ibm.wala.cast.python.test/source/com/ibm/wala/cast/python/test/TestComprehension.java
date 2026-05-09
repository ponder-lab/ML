package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.SSAContextInterpreter;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestComprehension extends TestJythonCallGraphShape {

  protected static final List<GraphAssertion> assertionsComp1 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script comp1.py"}),
          new GraphAssertion(
              "script comp1.py",
              new String[] {
                "CodeBody:__Lscript comp1.py/comprehension1",
                "CodeBody:__Lscript comp1.py/comprehension3"
              }),
          new GraphAssertion(
              "CodeBody:__Lscript comp1.py/comprehension1",
              new String[] {"script comp1.py/comprehension1"}),
          new GraphAssertion(
              "CodeBody:__Lscript comp1.py/comprehension3",
              new String[] {"script comp1.py/comprehension3"}),
          new GraphAssertion(
              "script comp1.py/comprehension1",
              new String[] {"script comp1.py/f1", "script comp1.py/f2", "script comp1.py/f3"}),
          new GraphAssertion(
              "script comp1.py/comprehension3",
              new String[] {
                "script comp1.py/f1/lambda1",
                "script comp1.py/f2/lambda1",
                "script comp1.py/f3/lambda1"
              }));

  @Test
  public void testComp1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = this.makeEngine("comp1.py");
    PropagationCallGraphBuilder callGraphBuilder = engine.defaultCallGraphBuilder();
    CallGraph CG = callGraphBuilder.makeCallGraph(callGraphBuilder.getOptions());

    /*
    CAstCallGraphUtil.AVOID_DUMP.set(false);
     CAstCallGraphUtil.dumpCG(
         (SSAContextInterpreter) callGraphBuilder.getContextInterpreter(),
         callGraphBuilder.getPointerAnalysis(),
         CG);
         */
    System.err.println(CG);
    verifyGraphAssertions(CG, assertionsComp1);
  }

  protected static final List<GraphAssertion> assertionsComp3 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script comp3.py"}),
          new GraphAssertion(
              "script comp3.py",
              new String[] {
                "CodeBody:__Lscript comp3.py/comprehension1",
                "CodeBody:__Lscript comp3.py/comprehension3"
              }),
          new GraphAssertion(
              "CodeBody:__Lscript comp3.py/comprehension1",
              new String[] {"script comp3.py/comprehension1"}),
          new GraphAssertion(
              "CodeBody:__Lscript comp3.py/comprehension3",
              new String[] {"script comp3.py/comprehension3"}),
          new GraphAssertion(
              "script comp3.py/comprehension1",
              new String[] {
                "script comp3.py/g1",
                "script comp3.py/g2",
                "script comp3.py/f1",
                "script comp3.py/f2",
                "script comp3.py/f3",
                "script comp3.py/g2/lambda1"
              }),
          new GraphAssertion(
              "script comp3.py/g2/lambda1",
              new String[] {"script comp3.py/f1", "script comp3.py/f2", "script comp3.py/f3"}),
          new GraphAssertion(
              "script comp3.py/comprehension3",
              new String[] {
                "script comp3.py/f1/lambda1",
                "script comp3.py/f2/lambda1",
                "script comp3.py/f3/lambda1"
              }));

  @Test
  public void testComp3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = this.makeEngine("comp3.py");
    PropagationCallGraphBuilder callGraphBuilder = engine.defaultCallGraphBuilder();
    CallGraph CG = callGraphBuilder.makeCallGraph(callGraphBuilder.getOptions());

    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) callGraphBuilder.getContextInterpreter(),
        callGraphBuilder.getPointerAnalysis(),
        CG);

    verifyGraphAssertions(CG, assertionsComp3);
  }
}
