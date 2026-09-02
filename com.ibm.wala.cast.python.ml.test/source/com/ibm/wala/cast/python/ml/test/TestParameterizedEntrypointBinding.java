package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Witnesses for wala/ML#868 layer two: a test-method parameter whose only supplier is a {@code
 * parameterized.parameters} decorator argument receives the named script-defined class objects
 * through the pytest entrypoint, one invocation per argument.
 *
 * <p>The assertions pin four properties. The bound parameter's points-to members are the defining
 * script's OWN class objects (asserted by concrete type, which only the module's allocation can
 * supply). Each parameterization gets its own context: some context holds exactly {@code A} and
 * another exactly {@code B}, never a union, because a single unioned binding would reintroduce the
 * receiver blur wala/ML#868 is about one level up. A decorator of identical shape but an
 * unrecognized name binds nothing, pinning the name gate: without it, this witness fails by the
 * unrecognized method's parameter acquiring a script class. And a from-imported bare-name argument
 * resolves through the unique cross-script match to the same defining-script object.
 */
public class TestParameterizedEntrypointBinding extends TestPythonMLCallGraphShape {

  private static final String A_CLASS = "Lscript parameterized_classes.py/A";

  private static final String B_CLASS = "Lscript parameterized_classes.py/B";

  @Test
  public void testDecoratorArgumentsReachTheParameter() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            Collections.<File>emptyList(),
            "test_parameterized_classes.py",
            "parameterized_classes.py",
            "parameterized.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    assertEquals(
        "Expecting each parameterization in its own context: exactly the two classes, each alone"
            + " in some context, never unioned.",
        Set.of(Set.of(A_CLASS), Set.of(B_CLASS)),
        boundParameterTypeSets(builder, cg, "test_pick"));

    assertEquals(
        "Expecting the unrecognized decorator to bind nothing: the binding is name-gated, and a"
            + " decorator of identical shape but different identity must not inject values.",
        Set.of(),
        boundParameterTypeSets(builder, cg, "test_unrecognized"));

    assertEquals(
        "Expecting a from-imported bare-name argument to resolve through the unique cross-script"
            + " match to the defining script's class object.",
        Set.of(Set.of(A_CLASS)),
        boundParameterTypeSets(builder, cg, "test_bare"));
  }

  /**
   * Collects, for each context of the named test method in which the class parameter points to at
   * least one script-defined class of the fixture's module, that context's set of such class type
   * names. Contexts whose parameter holds no such class (the script-driven raw call, the
   * fake-root's placeholder allocation) contribute nothing, so the result isolates exactly what the
   * entrypoint binding supplied.
   *
   * @param builder The propagation builder whose pointer analysis to consult.
   * @param cg The call graph.
   * @param method The test method's name.
   * @return The per-context sets of bound fixture-class type names.
   */
  private static Set<Set<String>> boundParameterTypeSets(
      PythonSSAPropagationCallGraphBuilder builder, CallGraph cg, String method) {
    Set<Set<String>> ret = new HashSet<>();

    for (CGNode node : cg) {
      String signature = node.getMethod().getSignature();
      if (!signature.contains("TestPick." + method + ".do")) continue;

      IR ir = node.getIR();
      if (ir == null || ir.getNumberOfParameters() < 3) continue;

      PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 3);
      Set<String> types = new TreeSet<>();

      for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(pk)) {
        String type = ik.concreteType().getName().toString();
        if (type.startsWith("Lscript parameterized_classes.py/")) types.add(type);
      }

      if (!types.isEmpty()) ret.add(types);
    }

    return ret;
  }

  /**
   * The bound parameter's receivers dispatch: the parameterized test method reaches both classes'
   * bodies, which is the call-edge presence wala/ML#868 reports missing.
   */
  @Test
  public void testBoundReceiversDispatch() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            Collections.<File>emptyList(),
            "test_parameterized_classes.py",
            "parameterized_classes.py",
            "parameterized.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    Set<String> successors = new HashSet<>();

    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().contains("TestPick.test_pick.do")) continue;
      for (java.util.Iterator<CGNode> it = cg.getSuccNodes(node); it.hasNext(); )
        successors.add(it.next().getMethod().getSignature());
    }

    for (String expected :
        List.of("parameterized_classes.py.A.do()LRoot;", "parameterized_classes.py.B.do()LRoot;"))
      assertTrue(
          "Expecting the parameterized method to reach: " + expected + " among: " + successors,
          successors.stream().anyMatch(s -> s.endsWith(expected)));
  }
}
