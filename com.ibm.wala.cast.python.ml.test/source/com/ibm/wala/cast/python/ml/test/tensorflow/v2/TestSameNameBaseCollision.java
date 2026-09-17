package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * A base written as a bare imported name must resolve to the class the import names, not to
 * whichever same-named class was parsed last (wala/ML#946). {@code tf2_657_model_call.py} declares
 * {@code from tensorflow.keras import Model; class MyModel(Model)} and {@code tf2_657_collide.py}
 * declares its own {@code class Model}; the superclass must be the summary shell in either module
 * order, where the unscoped lookup gave the colliding class in one order and {@code object} in the
 * other.
 */
public class TestSameNameBaseCollision extends AbstractTensorTest {

  private static final String SUBCLASS = "Lscript tf2_657_model_call.py/MyModel";

  private static final String SHELL = "Ltensorflow/keras/Model";

  @Test
  public void testCollidingModuleFirst()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertShellSuperclass(false);
  }

  @Test
  public void testSubclassModuleFirst()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertShellSuperclass(true);
  }

  private void assertShellSuperclass(boolean subclassFirst)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<String> names = new ArrayList<>(List.of("tf2_657_collide.py", "tf2_657_model_call.py"));
    if (subclassFirst) Collections.reverse(names);
    Set<Module> modules = new LinkedHashSet<>();
    for (String n : names) modules.add(getScript(n));
    PythonTensorAnalysisEngine engine =
        new PythonTensorAnalysisEngine(
            List.of(),
            PythonTensorAnalysisEngine.TENSORFLOW,
            PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
    engine.setModuleFiles(modules);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    IClass my =
        CG.getClassHierarchy()
            .lookupClass(TypeReference.findOrCreate(PythonTypes.pythonLoader, SUBCLASS));
    String order = subclassFirst ? "subclass module first" : "colliding module first";
    assertEquals(
        "The bare imported base must resolve to the shell, " + order + " (wala/ML#946).",
        SHELL,
        my.getSuperclass().getName().toString());
    assertTrue(
        "MyModel.call must keep its call-graph node (wala/ML#657).",
        CG.stream()
            .anyMatch(
                n ->
                    n.getMethod()
                        .getSignature()
                        .contains("tf2_657_model_call.py.MyModel.call.do")));
  }
}
