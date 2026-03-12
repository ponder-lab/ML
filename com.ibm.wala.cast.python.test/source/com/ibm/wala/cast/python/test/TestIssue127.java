package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Test;

/**
 * Isolated test case for https://github.com/wala/ML/issues/127.
 *
 * <p>This test verifies that WALA can resolve implicit {@code __call__} method invocations for
 * synthetic classes defined in XML summaries.
 */
public class TestIssue127 extends TestJythonCallGraphShape {

  private static Logger logger = Logger.getLogger(TestIssue127.class.getName());

  @Override
  protected PythonAnalysisEngine<?> createEngine(List<File> pythonPath)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    return new PythonAnalysisEngine<Void>(pythonPath) {
      @Override
      public PythonSSAPropagationCallGraphBuilder defaultCallGraphBuilder()
          throws IllegalArgumentException {
        try {
          PythonSSAPropagationCallGraphBuilder builder = super.defaultCallGraphBuilder();
          addSummaryBypassLogic(getOptions(), "issue127.xml");
          return builder;
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
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
   * @see <a href="https://github.com/wala/ML/issues/127">Issue 127</a>
   */
  @Test
  public void testIssue127()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("issue127.py");
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
      if (node.getMethod().getDeclaringClass().getName().toString().equals("Lscript issue127.py")) {
        for (Iterator<CGNode> it = CG.getSuccNodes(node); it.hasNext(); ) {
          CGNode callee = it.next();
          logger.info("Found callee: " + callee.getMethod().getSignature());
          if (callee.getMethod().getDeclaringClass().getName().toString().equals("Lissue127/C")) {
            // check if it calls __call__
            for (Iterator<CGNode> it2 = CG.getSuccNodes(callee); it2.hasNext(); ) {
              CGNode callee2 = it2.next();
              logger.info("Found callee2: " + callee2.getMethod().getSignature());
            }
          }
          if (callee.getMethod().getDeclaringClass().getName().toString().equals("Lissue127/C")
              && callee.getMethod().getName().toString().equals("__call__")) {
            found = true;
          }
        }
      }
    }

    // FIXME: Change to assertTrue once https://github.com/wala/ML/issues/127 is fixed.
    assertFalse("Expecting to find __call__ method trampoline.", found);
  }
}
