package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Iterator;
import org.junit.Test;

/**
 * The comprehension trampoline reads a comprehension invoke's iterables from its positional
 * parameters, so the invoke's positional count and its total use count must agree wherever the
 * trampoline is the callee (<a href="https://github.com/wala/ML/issues/917">wala/ML#917</a>).
 *
 * <p>Today they agree by construction: {@code PythonCAstToIRTranslator} emits every comprehension
 * invoke with an empty keyword list, so this test cannot fail on the current front end, and it is
 * here as the tripwire for the change that makes it fail on purpose. When a comprehension's {@code
 * if} filters are carried into the IR as keyword parameters (the remedy for wala/ML#917), the two
 * counts stop agreeing, and this test must then be revised as part of that change to assert that
 * the trampoline reads the positional count only. A future failure here means a keyword use reached
 * a comprehension invoke; it is not a stale test to delete.
 */
public class TestComprehensionInvokeArity extends TestJythonCallGraphShape {

  private static final String[] FIXTURES = {
    "comp1.py", "comp2.py", "comp3.py", "comp4.py", "comp5.py", "comp6.py"
  };

  @Test
  public void testComprehensionInvokesCarryNoKeywordParameters()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    int seen = 0;
    for (String fixture : FIXTURES) {
      PythonAnalysisEngine<?> engine = this.makeEngine(fixture);
      PropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      CallGraph cg = builder.makeCallGraph(builder.getOptions());
      IClass comprehension = cg.getClassHierarchy().lookupClass(PythonTypes.comprehension);
      for (CGNode node : cg) {
        if (node.getIR() == null) continue;
        for (Iterator<CallSiteReference> sites = node.iterateCallSites(); sites.hasNext(); ) {
          CallSiteReference site = sites.next();
          boolean toComprehension = false;
          for (CGNode target : cg.getPossibleTargets(node, site))
            if (cg.getClassHierarchy()
                .isSubclassOf(target.getMethod().getDeclaringClass(), comprehension))
              toComprehension = true;
          if (!toComprehension) continue;
          for (SSAAbstractInvokeInstruction call : node.getIR().getCalls(site)) {
            if (!(call instanceof PythonInvokeInstruction)) continue;
            PythonInvokeInstruction invoke = (PythonInvokeInstruction) call;
            assertEquals(
                fixture + ": a comprehension invoke carries a keyword parameter: " + invoke,
                invoke.getNumberOfPositionalParameters(),
                invoke.getNumberOfUses());
            seen++;
          }
        }
      }
    }
    assertTrue(
        "no comprehension invoke was found in the fixtures; the check ran on nothing", seen > 0);
  }
}
