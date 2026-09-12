package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.Test;

/**
 * A comprehension invoke carries its iterables as positional parameters and its {@code if} filters
 * as keyword parameters, one per filter, and the comprehension trampoline reads it that way: one
 * element iteration per positional iterable and one filter-body callee per keyword (<a
 * href="https://github.com/wala/ML/issues/917">wala/ML#917</a>).
 *
 * <p>The filter-body edge is the assertion that tells a bound keyword from a present one: the
 * builder binds a keyword to the callee's parameter of the same local name and silently drops a
 * keyword the callee does not name, so a trampoline that carried the filters as unnamed slots would
 * still show the keyword on the invoke while no filter body reached the call graph.
 */
public class TestComprehensionInvokeArity extends TestJythonCallGraphShape {

  /**
   * Per fixture, the filter count of each comprehension invoke in the file, in ascending order: the
   * comprehensions of {@code comp1.py} and {@code comp3.py} carry one {@code if} apiece beside an
   * unfiltered one, and {@code comp7.py} adds a comprehension with two {@code if} clauses (two
   * distinct filter functions, so the loader must not fold them into one class) and a filtered
   * comprehension as a method's default value, which the parser visits in the class's own scope.
   * That last comprehension has no invoke to count: its element lambda is declared under the class
   * while the default's code runs in the script, where the function expression resolves to a
   * redefinition twin with no body, so the invoke never reaches a trampoline (<a
   * href="https://github.com/wala/ML/issues/918">wala/ML#918</a>, older than the filters); what it
   * witnesses here is the filter's type, checked below. A dict comprehension ({@code comp2.py},
   * {@code comp5.py}, {@code comp6.py}) is lowered to a loop by the parser and carries no
   * comprehension invoke.
   */
  private static final Map<String, List<Integer>> FILTERS_PER_COMPREHENSION =
      Map.of(
          "comp1.py", List.of(0, 1),
          "comp2.py", List.of(),
          "comp3.py", List.of(0, 1),
          "comp4.py", List.of(0),
          "comp5.py", List.of(),
          "comp6.py", List.of(),
          "comp7.py", List.of(0, 0, 0, 0, 2));

  /** The type-name shape of the front-end's filter functions, with any redefinition suffix. */
  private static final Pattern FILTER_CLASS = Pattern.compile(".*/filter\\d+(\\$\\d+)?");

  @Test
  public void testComprehensionInvokesCarryTheirFiltersAsKeywords()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    for (Map.Entry<String, List<Integer>> expected : FILTERS_PER_COMPREHENSION.entrySet()) {
      String fixture = expected.getKey();
      PythonAnalysisEngine<?> engine = this.makeEngine(fixture);
      PropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      CallGraph cg = builder.makeCallGraph(builder.getOptions());
      IClassHierarchy cha = cg.getClassHierarchy();
      IClass comprehension = cha.lookupClass(PythonTypes.comprehension);
      IClass filter = cha.lookupClass(PythonTypes.filter);
      List<Integer> filterCounts = new ArrayList<>();
      List<String> seen = new ArrayList<>();

      // A filter is a function of its own wherever the comprehension is built. In a class's own
      // scope (a method's default value, comp7.py's Holder.ws) the parser would otherwise take the
      // filter for a method of the class, which drops it from the comprehension invoke.
      int classScopeFilters = 0;
      for (IClass c : cha) {
        String name = c.getName().toString();
        if (!FILTER_CLASS.matcher(name).matches()) continue;
        assertEquals(
            fixture + ": " + name + " is not a filter function", filter, c.getSuperclass());
        if (name.contains("/Holder/")) classScopeFilters++;
      }
      assertEquals(
          fixture + ": class-scope filters", fixture.equals("comp7.py") ? 1 : 0, classScopeFilters);
      for (CGNode node : cg) {
        if (node.getIR() == null) continue;
        for (Iterator<CallSiteReference> sites = node.iterateCallSites(); sites.hasNext(); ) {
          CallSiteReference site = sites.next();
          // The trampoline is the summarized function declared on the comprehension's class; the
          // element lambda's body, which the trampoline itself invokes, is declared there too but
          // is real code, not a comprehension invoke.
          Set<CGNode> trampolines = new HashSet<>();
          for (CGNode target : cg.getPossibleTargets(node, site))
            if (target.getMethod() instanceof PythonSummarizedFunction
                && cha.isSubclassOf(target.getMethod().getDeclaringClass(), comprehension))
              trampolines.add(target);
          if (trampolines.isEmpty()) continue;
          for (SSAAbstractInvokeInstruction call : node.getIR().getCalls(site)) {
            if (!(call instanceof PythonInvokeInstruction)) continue;
            PythonInvokeInstruction invoke = (PythonInvokeInstruction) call;
            int positional = invoke.getNumberOfPositionalParameters();
            int keywords = invoke.getNumberOfKeywordParameters();
            String where = fixture + ": " + invoke;
            assertEquals(where, positional + keywords, invoke.getNumberOfUses());

            // Every keyword value is an allocation of a filter function.
            for (int k = 0; k < keywords; k++) {
              SSAInstruction def = node.getDU().getDef(invoke.getUse(positional + k));
              assertTrue(
                  where + ": keyword " + k + " is not a filter allocation: " + def,
                  def instanceof SSANewInstruction
                      && cha.isSubclassOf(
                          cha.lookupClass(((SSANewInstruction) def).getConcreteType()), filter));
            }

            // The trampoline iterates exactly the positional iterables (the lambda and the fresh
            // collection precede them) and calls exactly one distinct filter body per keyword.
            for (CGNode trampoline : trampolines) {
              int iterations = 0;
              for (SSAInstruction inst : trampoline.getIR().getInstructions())
                if (inst instanceof EachElementGetInstruction) iterations++;
              assertEquals(
                  where + ": iterables read by the trampoline", positional - 2, iterations);

              // Distinct METHODS, not nodes: two filters folded into one class are one method
              // reached from two call sites, which context sensitivity shows as two nodes.
              Set<IMethod> filterBodies = new HashSet<>();
              for (Iterator<CallSiteReference> inner = trampoline.iterateCallSites();
                  inner.hasNext(); ) {
                for (CGNode callee : cg.getPossibleTargets(trampoline, inner.next()))
                  if (cha.isSubclassOf(callee.getMethod().getDeclaringClass(), filter))
                    filterBodies.add(callee.getMethod());
              }
              assertEquals(
                  where + ": filter bodies called by the trampoline " + filterBodies,
                  keywords,
                  filterBodies.size());
            }

            filterCounts.add(keywords);
            seen.add(node.getMethod().getDeclaringClass().getName() + " -> " + keywords);
          }
        }
      }
      Collections.sort(filterCounts);
      assertEquals(
          fixture + ": filters per comprehension " + seen, expected.getValue(), filterCounts);
    }
  }
}
