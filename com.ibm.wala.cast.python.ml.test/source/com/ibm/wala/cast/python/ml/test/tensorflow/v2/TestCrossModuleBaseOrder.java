package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A user base class imported from another module must resolve whatever the module order
 * (wala/ML#944). The fixture's subclass module sorts before its base module, so an ascending path
 * order translates the subclasses first; on the unfixed engine that order drops all three
 * subclasses to {@code object} and the inherited method gets no call-graph node, while the
 * descending order resolves them. The three subclasses sit at three nesting depths (top level,
 * inside a function, inside a class), each driven with a distinct shape, so the inherited method's
 * parameter says which of them resolved.
 */
public class TestCrossModuleBaseOrder extends AbstractTensorTest {

  private static final String PROJECT = "order_proj";

  private static final String[] FILES = {
    "order_proj/alpha_sub.py", "order_proj/driver.py", "order_proj/zeta_base.py"
  };

  private static final String BASE = "Lscript zeta_base.py/Base";

  private static final String[] SUBCLASSES = {
    "Lscript alpha_sub.py/Sub",
    "Lscript alpha_sub.py/scale_inside/Inner",
    "Lscript alpha_sub.py/Outer/Nested"
  };

  /** The inherited method's parameter carries all three drivers' shapes, one per nesting depth. */
  @Test
  public void testInheritedMethodParameterAcrossNestings()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "zeta_base.py",
        "Base.scale",
        PROJECT,
        1,
        2,
        Map.of(
            3,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3),
                TensorType.of(FLOAT_32, 4),
                TensorType.of(FLOAT_32, 5, 6, 7))));
  }

  @Test
  public void testBaseResolvesInAscendingOrder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertResolved(false);
  }

  @Test
  public void testBaseResolvesInDescendingOrder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertResolved(true);
  }

  private void assertResolved(boolean descending)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // The fixture is read from the class path, as the harness reads it, so the arms hold wherever
    // the packaged copy lives; only the insertion order of the modules differs between the arms.
    List<String> names = new ArrayList<>(List.of(FILES));
    if (descending) Collections.reverse(names);
    Set<Module> modules = new LinkedHashSet<>();
    for (String name : names) modules.add(getScript(name));
    PythonTensorAnalysisEngine engine =
        new PythonTensorAnalysisEngine(
            this.getPathFiles(PROJECT),
            PythonTensorAnalysisEngine.TENSORFLOW,
            PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
    engine.setModuleFiles(modules);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    IClassHierarchy cha = CG.getClassHierarchy();
    String order = descending ? "descending" : "ascending";
    for (String sub : SUBCLASSES) {
      IClass cls = cha.lookupClass(TypeReference.findOrCreate(PythonTypes.pythonLoader, sub));
      assertEquals(
          sub + " must extend the imported base in " + order + " module order (wala/ML#944).",
          BASE,
          cls.getSuperclass().getName().toString());
    }
    MethodReference scale =
        MethodReference.findOrCreate(
            TypeReference.findOrCreate(PythonTypes.pythonLoader, BASE + "/scale"),
            AstMethodReference.fnSelector);
    assertFalse(
        "The inherited method must have call-graph nodes in " + order + " module order.",
        CG.getNodes(scale).isEmpty());
  }

  /**
   * Same-module twin of the function-nested arm: no module order involved. A class defined inside a
   * function now declares its name in that scope and the inherited-member propagation reads the
   * base by its global name, so the inherited method dispatches (wala/ML#945); before the fix this
   * read {@code Function must exist in call graph}.
   */
  @Test
  public void testFunctionNestedSubclassSameModule()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "order_nested_local.py", "Base.scale", 1, 2, Map.of(3, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The one shape a base-dependency cycle can take and still run: {@code a} takes {@code A}'s base
   * from {@code b} at module level, and {@code b} takes {@code C}'s base from {@code a} inside a
   * function body, because a module-level import in both directions leaves one module partially
   * initialised and Python refuses the program. Dependency-ordered translation cannot put both
   * modules first, so it falls back to the given order and one of the two bases resolves late. This
   * arm is the standing test of the premise the ordering rests on: it fails while cycles are
   * unorderable, and would pass only if a cycle were resolved some other way. TODO: a documented
   * limit of wala/ML#944's fix, reachable only through a base import deferred into a function.
   */
  @Test(expected = AssertionError.class)
  public void testCyclicBaseDependencyFallsBackToTheGivenOrder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    for (boolean descending : new boolean[] {false, true}) {
      List<String> names =
          new ArrayList<>(List.of("cyc_proj/a.py", "cyc_proj/b.py", "cyc_proj/driver.py"));
      if (descending) Collections.reverse(names);
      Set<Module> modules = new LinkedHashSet<>();
      for (String name : names) modules.add(getScript(name));
      PythonTensorAnalysisEngine engine =
          new PythonTensorAnalysisEngine(
              this.getPathFiles("cyc_proj"),
              PythonTensorAnalysisEngine.TENSORFLOW,
              PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
      engine.setModuleFiles(modules);
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      addPytestEntrypoints(builder);
      CallGraph CG = builder.makeCallGraph(builder.getOptions());
      IClassHierarchy cha = CG.getClassHierarchy();
      assertEquals(
          "Lscript b.py/B",
          cha.lookupClass(TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lscript a.py/A"))
              .getSuperclass()
              .getName()
              .toString());
      assertEquals(
          "Lscript a.py/A",
          cha.lookupClass(
                  TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lscript b.py/make/C"))
              .getSuperclass()
              .getName()
              .toString());
    }
  }
}
