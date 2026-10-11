package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
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
 * A class whose base is a class a package re-exports, {@code class C(pkg.Base)} or {@code from pkg
 * import Base; class C(Base)} where {@code pkg/__init__.py} imports {@code Base} from the module
 * that defines it, inherits that base (<a href="https://github.com/wala/ML/issues/1042">
 * wala/ML#1042</a>). The base was looked up only among the classes the named module defines, and a
 * package's initialization module defines none of the classes it re-exports, so the class fell to
 * {@code object}: the base's initializer never ran on its instances, the fields it sets were never
 * written, and a call on an instance did not dispatch to the inherited layer {@code call}.
 */
public class TestReexportedBaseClass extends AbstractTensorTest {

  private static final String[] FILES = {
    "reexport_base_proj/pkg/__init__.py",
    "reexport_base_proj/pkg/inputters/__init__.py",
    "reexport_base_proj/pkg/inputters/inputter.py",
    "reexport_base_proj/pkg/models/__init__.py",
    "reexport_base_proj/pkg/models/s2s.py",
    "reexport_base_proj/vendor/__init__.py",
    "reexport_base_proj/vendor/pkg/__init__.py",
    "reexport_base_proj/vendor/pkg/inputters/__init__.py",
    "reexport_base_proj/vendor/pkg/inputters/inputter.py",
    "reexport_base_proj/main.py"
  };

  /** A package and one of its modules that depend on each other for translation order. */
  private static final String[] CYCLE_FILES = {
    "reexport_cycle_proj/aaa_waiter.py",
    "reexport_cycle_proj/cyc/__init__.py",
    "reexport_cycle_proj/cyc/base.py",
    "reexport_cycle_proj/cyc/mod.py",
    "reexport_cycle_proj/main.py"
  };

  private static final String CYCLE_PROJECT = "reexport_cycle_proj";

  private static final String PROJECT = "reexport_base_proj";

  private static final Map<Integer, Set<TensorType>> FLOAT_2_3 =
      Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3)));

  /** The base imported from the module that defines it, the control. */
  @Test
  public void testBaseFromDefiningModule()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_defining_module", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * The base written as an attribute of the re-exporting package, {@code class
   * C(inputters.ExampleInputter)}: the field the base's initializer sets is read on the subclass's
   * instance.
   */
  @Test
  public void testBaseThroughPackageAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_package_attribute", PROJECT, 1, 1, FLOAT_2_3);
  }

  /** The base imported by name from the re-exporting package, {@code from pkg.inputters import}. */
  @Test
  public void testBaseThroughPackageName()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_package_name", PROJECT, 1, 1, FLOAT_2_3);
  }

  /** The base re-exported by one package from another that re-exports it, two hops. */
  @Test
  public void testBaseThroughTwoPackages()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_two_packages", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * A call on an instance of the subclass dispatches to the layer {@code call} it inherits through
   * the re-exported base.
   */
  @Test
  public void testLayerCallThroughPackageAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_layer_call", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * The base re-exported by a relative import in the package's initialization module, {@code from
   * .inputter import Inputter}.
   */
  @Test
  public void testBaseThroughRelativeReexport()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_relative_reexport", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * The project's own {@code pkg.inputters} is chosen over a vendored copy at {@code
   * vendor/pkg/inputters} whose path also ends in the module's path and whose base sets no
   * inputter: every arm above runs with the copy in scope, and this one names the collision.
   */
  @Test
  public void testOwnPackageOverVendoredCopy()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_package_attribute", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * A package that re-exports a class from one of its modules, {@code cyc/__init__.py} from {@code
   * cyc/mod.py}, while that module takes its own base through the package: the two depend on each
   * other for translation order. A class in a third module that only waits behind the cycle, {@code
   * aaa_waiter.py}'s {@code Waiter}, keeps its order, and both classes get their bases in either
   * module order.
   */
  @Test
  public void testCycleKeepsTheOrderOfClassesBehindIt() throws Exception {
    for (boolean descending : new boolean[] {false, true}) {
      List<String> names = new ArrayList<>(List.of(CYCLE_FILES));
      if (descending) Collections.reverse(names);
      Set<Module> modules = new LinkedHashSet<>();
      for (String name : names) modules.add(getScript(name));
      PythonTensorAnalysisEngine engine =
          new PythonTensorAnalysisEngine(
              this.getPathFiles(CYCLE_PROJECT),
              PythonTensorAnalysisEngine.TENSORFLOW,
              PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
      engine.setModuleFiles(modules);
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      CallGraph cg = builder.makeCallGraph(builder.getOptions());
      IClassHierarchy cha = cg.getClassHierarchy();
      String order = descending ? "descending" : "ascending";
      assertEquals(
          "Waiter's base in " + order + " module order",
          "Lscript cyc/mod.py/Child",
          superclassOf(cha, "Lscript aaa_waiter.py/Waiter"));
      assertEquals(
          "Child's base in " + order + " module order",
          "Lscript cyc/base.py/Base",
          superclassOf(cha, "Lscript cyc/mod.py/Child"));
    }
  }

  /** The value the waiting class's layer call returns, through the cycle. */
  @Test
  public void testLayerCallBehindCycle()
      throws ClassHierarchyException, CancelException, IOException {
    test(CYCLE_FILES, "main.py", "consume", CYCLE_PROJECT, 1, 1, FLOAT_2_3);
  }

  private static String superclassOf(IClassHierarchy cha, String name) {
    IClass cls = cha.lookupClass(TypeReference.findOrCreate(PythonTypes.pythonLoader, name));
    assertNotNull(name + " must be a class", cls);
    return cls.getSuperclass().getName().toString();
  }
}
