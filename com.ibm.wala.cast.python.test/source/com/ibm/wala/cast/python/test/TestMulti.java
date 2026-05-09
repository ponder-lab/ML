package com.ibm.wala.cast.python.test;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.util.test.TestCallGraphShape.GraphAssertion;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.SSAContextInterpreter;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.NullProgressMonitor;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

public class TestMulti extends TestJythonCallGraphShape {

  protected static final List<GraphAssertion> assertionsCalls1 =
      List.of(new GraphAssertion(ROOT, new String[] {"script calls1.py", "script calls2.py"}));

  @Test
  public void testCalls1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("calls1.py", "calls2.py");
    verifyGraphAssertions(CG, assertionsCalls1);
  }

  protected static final List<GraphAssertion> assertionsMulti1 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script multi1.py", "script multi2.py"}),
          new GraphAssertion("script multi1.py", new String[] {"script multi2.py/silly"}),
          new GraphAssertion(
              "script multi2.py/silly", new String[] {"script multi2.py/silly/inner"}));

  @Test
  public void testMulti1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("multi2.py", "multi1.py");
    PropagationCallGraphBuilder builder =
        (PropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(engine.getOptions(), new NullProgressMonitor());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    verifyGraphAssertions(CG, assertionsMulti1);
  }

  protected static final List<GraphAssertion> assertionsMulti2 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script multi1.py", "script multi2.py"})
          // TODO: Add the following code once https://github.com/wala/ML/issues/168 is fixed:
          // new GraphAssertion("script multi1.py", new String[] {"script multi2.py/silly"})
          // TODO: Add the following code once https://github.com/wala/ML/issues/168 is fixed:
          // new GraphAssertion("script multi2.py/silly", new String[] {"script
          // multi2.py/silly/inner"})
          );

  @Test
  public void testMulti2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("multi1.py", "multi2.py");
    PropagationCallGraphBuilder builder =
        (PropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(engine.getOptions(), new NullProgressMonitor());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    verifyGraphAssertions(CG, assertionsMulti2);
  }

  protected static final List<GraphAssertion> assertionsMulti3 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script multi3.py", "script multi2.py"}),
          new GraphAssertion("script multi3.py", new String[] {"script multi2.py/silly"}),
          new GraphAssertion(
              "script multi2.py/silly", new String[] {"script multi2.py/silly/inner"}));

  @Test
  public void testMulti3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("multi2.py", "multi3.py");
    PropagationCallGraphBuilder builder =
        (PropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(engine.getOptions(), new NullProgressMonitor());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    verifyGraphAssertions(CG, assertionsMulti3);
  }

  protected static final List<GraphAssertion> assertionsMulti4 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script multi4.py", "script multi2.py"}),
          new GraphAssertion("script multi4.py", new String[] {"script multi2.py/silly"}),
          new GraphAssertion(
              "script multi2.py/silly", new String[] {"script multi2.py/silly/inner"}));

  @Test
  public void testMulti4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("multi2.py", "multi4.py");
    PropagationCallGraphBuilder builder =
        (PropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(engine.getOptions(), new NullProgressMonitor());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    verifyGraphAssertions(CG, assertionsMulti4);
  }

  protected static final List<GraphAssertion> assertionsMulti5 =
      List.of(
          new GraphAssertion(ROOT, new String[] {"script multi5.py", "script multi2.py"}),
          new GraphAssertion("script multi5.py", new String[] {"script multi2.py/silly"}),
          new GraphAssertion(
              "script multi2.py/silly", new String[] {"script multi2.py/silly/inner"}));

  @Test
  public void testMulti5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("multi2.py", "multi5.py");
    PropagationCallGraphBuilder builder =
        (PropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(engine.getOptions(), new NullProgressMonitor());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    verifyGraphAssertions(CG, assertionsMulti5);
  }
}
