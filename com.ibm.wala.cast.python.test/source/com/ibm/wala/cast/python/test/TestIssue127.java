package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Test;

/** Isolated test case for https://github.com/wala/ML/issues/127. */
public class TestIssue127 extends TestJythonCallGraphShape {

  private static Logger logger = Logger.getLogger(TestIssue127.class.getName());

  @Override
  protected PythonAnalysisEngine<?> createEngine(List<File> pythonPath)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    return new PythonAnalysisEngine<Void>(pythonPath) {
      @Override
      protected void addBypassLogic(IClassHierarchy cha, AnalysisOptions options) {
        super.addBypassLogic(cha, options);
        addSummaryBypassLogic(options, "issue127.xml");
        addSummaryBypassLogic(options, "issue127b.xml");
        addSummaryBypassLogic(options, "issue127c.xml");
        addSummaryBypassLogic(options, "issue127d.xml");
      }

      @Override
      public Void performAnalysis(PropagationCallGraphBuilder builder) throws CancelException {
        assert false;
        return null;
      }
    };
  }

  /**
   * Test implicit {@code __call__} on a synthetic object.
   *
   * <p>This test verifies that WALA can resolve implicit {@code __call__} method invocations for
   * synthetic classes defined in XML summaries.
   *
   * @see <a href="https://github.com/wala/ML/issues/127">Issue 127</a>
   */
  @Test
  public void testIssue127()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("test_issue127.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    if (logger.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      logger.fine("Call graph:\n" + CG);
    }

    boolean found = false;

    for (CGNode node : CG) {
      if (node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .equals("Lscript test_issue127.py")) {
        for (Iterator<CGNode> it = CG.getSuccNodes(node); it.hasNext(); ) {
          CGNode callee = it.next();
          String calleeName = callee.getMethod().getDeclaringClass().getName().toString();
          logger.info("Found callee: " + calleeName + "." + callee.getMethod().getName());
          if ((calleeName.endsWith("/C")
                  || calleeName.contains("/C/")
                  || calleeName.contains("/C/__call__"))
              && callee.getMethod().getName().toString().matches("(__call__|trampoline.*)")) {
            found = true;
          }
        }
      }
    }

    assertTrue("Expecting to find __call__ method trampoline.", found);
  }

  /**
   * Test explicit method call on a synthetic object.
   *
   * @see <a href="https://github.com/wala/ML/issues/127">Issue 127</a>
   */
  @Test
  public void testIssue127b()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("test_issue127b.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    if (logger.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      logger.fine("Call graph:\n" + CG);
    }

    boolean found = false;

    for (CGNode node : CG) {
      if (node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .equals("Lscript test_issue127b.py")) {
        for (Iterator<CGNode> it = CG.getSuccNodes(node); it.hasNext(); ) {
          CGNode callee = it.next();
          String calleeName = callee.getMethod().getDeclaringClass().getName().toString();
          logger.info("Found callee: " + calleeName + "." + callee.getMethod().getName());
          if ((calleeName.endsWith("/C") || calleeName.contains("/C/"))
              && callee.getMethod().getName().toString().matches("(foo|trampoline.*)")) {
            found = true;
          }
        }
      }
    }

    assertTrue("Expecting to find foo method call.", found);
  }

  /**
   * Test explicit method call on a real (non-synthetic) object.
   *
   * <p>This is a control case for comparison with {@link #testIssue127b()}.
   */
  @Test
  public void testIssue127bReal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("test_issue127b_real.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    if (logger.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      logger.fine("Call graph:\n" + CG);
    }

    boolean found = false;

    for (CGNode node : CG) {
      if (node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .equals("Lscript test_issue127b_real.py")) {
        for (Iterator<CGNode> it = CG.getSuccNodes(node); it.hasNext(); ) {
          CGNode callee = it.next();
          String calleeName = callee.getMethod().getDeclaringClass().getName().toString();
          logger.info("Found callee: " + calleeName + "." + callee.getMethod().getName());
          if ((calleeName.endsWith("/C") || calleeName.contains("/C/"))
              && callee.getMethod().getName().toString().matches("(foo|trampoline.*)")) {
            found = true;
          }
        }
      }
    }

    assertTrue("Expecting to find foo method call.", found);
  }

  /**
   * Test precision when calling the same method on two different objects.
   *
   * @see <a href="https://github.com/wala/ML/issues/127">Issue 127</a>
   */
  @Test
  public void testIssue127c()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("test_issue127c.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    if (logger.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      logger.fine("Call graph:\n" + CG);
    }

    int foundCount = 0;
    Set<Object> argValues = new HashSet<>();
    for (CGNode node : CG) {
      if (node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .equals("Lscript test_issue127c.py")) {
        for (Iterator<CGNode> it = CG.getSuccNodes(node); it.hasNext(); ) {
          CGNode callee = it.next();
          String calleeName = callee.getMethod().getDeclaringClass().getName().toString();
          if ((calleeName.endsWith("/C") || calleeName.contains("/C/"))
              && callee.getMethod().getName().toString().matches("(foo|trampoline.*)")) {
            foundCount++;

            // Check parameter 2 (index 2 for positional param 'a', self is param 1)
            int[] params = callee.getIR().getSymbolTable().getParameterValueNumbers();
            if (params.length > 1) {
              PointerKey pk =
                  builder.getPointerKeyFactory().getPointerKeyForLocal(callee, params[1]);
              for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(pk)) {
                if (ik instanceof ConstantKey) {
                  argValues.add(((ConstantKey<?>) ik).getValue());
                }
              }
            }
          }
        }
      }
    }

    logger.fine("Test 127c argValues: " + argValues + " foundCount: " + foundCount);

    boolean has5 = false, has10 = false;

    for (Object v : argValues) {
      if (v instanceof Number) {
        if (((Number) v).intValue() == 5) has5 = true;
        if (((Number) v).intValue() == 10) has10 = true;
      }
    }

    assertTrue("Expecting to find foo method calls for both objects.", foundCount >= 2);
    assertTrue(
        "Expecting to find distinct arguments 5 and 10 tracked by pointer analysis.",
        has5 && has10);
  }

  /**
   * Test precision when calling the same method on an object directly and via variable.
   *
   * @see <a href="https://github.com/wala/ML/issues/127">Issue 127</a>
   */
  @Test
  public void testIssue127d()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("test_issue127d.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    if (logger.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      logger.fine("Call graph:\n" + CG);
    }

    int foundCount = 0;
    Set<Object> argValues = new HashSet<>();

    for (CGNode node : CG) {
      if (node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .equals("Lscript test_issue127d.py")) {
        for (Iterator<CGNode> it = CG.getSuccNodes(node); it.hasNext(); ) {
          CGNode callee = it.next();
          String calleeName = callee.getMethod().getDeclaringClass().getName().toString();
          if ((calleeName.endsWith("/C") || calleeName.contains("/C/"))
              && callee.getMethod().getName().toString().matches("(foo|trampoline.*)")) {
            foundCount++;

            int[] params = callee.getIR().getSymbolTable().getParameterValueNumbers();
            if (params.length > 1) {
              PointerKey pk =
                  builder.getPointerKeyFactory().getPointerKeyForLocal(callee, params[1]);
              for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(pk)) {
                if (ik instanceof ConstantKey) {
                  argValues.add(((ConstantKey<?>) ik).getValue());
                }
              }
            }
          }
        }
      }
    }

    logger.fine("Test 127d argValues: " + argValues + " foundCount: " + foundCount);

    boolean has5 = false, has3 = false;

    for (Object v : argValues) {
      if (v instanceof Number) {
        if (((Number) v).intValue() == 5) has5 = true;
        if (((Number) v).intValue() == 3) has3 = true;
      }
    }

    assertTrue(
        "Expecting to find foo method calls for direct and variable calls.", foundCount >= 2);
    assertTrue(
        "Expecting to find distinct arguments 5 and 3 tracked by pointer analysis.", has5 && has3);
  }
}
