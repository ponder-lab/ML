package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a program class subclassing a built-in Keras layer is a Keras layer in the class
 * hierarchy (wala/ML#1098): its superclass chain reaches {@code Ltensorflow/keras/layers/Layer}
 * through the built-in layer. Before, the built-in layer summaries declared no superclass, so they
 * had no class shell, the program class's base did not resolve, and its superclass was {@code
 * object}.
 */
public class TestBuiltinLayerSubclass extends AbstractTensorTest {

  private static final String FILE = "tf2_test_builtin_layer_subclass.py";

  /** The Keras layer base class. */
  private static final String LAYER = "Ltensorflow/keras/layers/Layer";

  /** A subclass of {@code BatchNormalization} that overrides {@code call}. */
  @Test
  public void testBatchNormalizationSubclass()
      throws ClassHierarchyException, CancelException, IOException {
    List<String> chain = this.superclassChain("BatchNormalization");
    assertTrue("The chain reaches Layer: " + chain, chain.contains(LAYER));
  }

  /** A subclass of {@code Dense} that adds nothing. */
  @Test
  public void testDenseSubclass() throws ClassHierarchyException, CancelException, IOException {
    List<String> chain = this.superclassChain("D");
    assertTrue("The chain reaches Layer: " + chain, chain.contains(LAYER));
  }

  /**
   * The declared bases of a subclass of {@code BatchNormalization}, walked transitively through
   * each base's own declared bases, reach {@code Layer}: the walk a client makes over {@link
   * IPythonClass#getBaseTypeNames()} to recognize a Keras layer.
   */
  @Test
  public void testBatchNormalizationSubclassBases()
      throws ClassHierarchyException, CancelException, IOException {
    Set<String> bases = this.declaredBaseClosure("BatchNormalization");
    assertTrue("The bases reach Layer: " + bases, bases.contains(LAYER));
  }

  /** As {@link #testBatchNormalizationSubclassBases()}, for a subclass of {@code Dense}. */
  @Test
  public void testDenseSubclassBases()
      throws ClassHierarchyException, CancelException, IOException {
    Set<String> bases = this.declaredBaseClosure("D");
    assertTrue("The bases reach Layer: " + bases, bases.contains(LAYER));
  }

  /**
   * A subclass of {@code Dense} that overrides {@code call}: its instances run the override, so the
   * override's input is the tensor the instance is called with. The override's {@code
   * super().call(inputs)} resolves to {@code Dense}'s {@code call} through the built-in layer's
   * class shell (wala/ML#1098), and that {@code call} is typed as {@code Dense}'s {@code __call__}
   * (wala/ML#1108), so the override's tensors are its input, the {@code super().call} result and
   * the product of that result with {@code 2.0}.
   */
  @Test
  public void testDenseSubclassOverride()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Scaled.call", 1, 3, Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * The names of a program class of the fixture and its superclasses, nearest first.
   *
   * @param name The program class's name in the fixture.
   * @return The class names along the superclass chain.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private List<String> superclassChain(String name)
      throws ClassHierarchyException, CancelException, IOException {
    List<String> ret = new ArrayList<>();
    for (IClass c = this.programClass(this.classHierarchy(), name);
        c != null;
        c = c.getSuperclass()) ret.add(c.getName().toString());
    return ret;
  }

  /**
   * The names of the declared bases of a program class of the fixture, and of their declared bases
   * in turn, each looked up in the class hierarchy. A class that records no bases answers with its
   * superclass, as {@link IPythonClass#getBaseTypeNames()} does by default.
   *
   * @param name The program class's name in the fixture.
   * @return The base names reached.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<String> declaredBaseClosure(String name)
      throws ClassHierarchyException, CancelException, IOException {
    IClassHierarchy cha = this.classHierarchy();
    Set<String> ret = new LinkedHashSet<>();
    Deque<IClass> work = new ArrayDeque<>(List.of(this.programClass(cha, name)));
    while (!work.isEmpty()) {
      IClass c = work.pop();
      List<TypeName> bases =
          c instanceof IPythonClass python
              ? python.getBaseTypeNames()
              : c.getSuperclass() == null ? List.of() : List.of(c.getSuperclass().getName());
      for (TypeName base : bases)
        if (ret.add(base.toString())) {
          IClass next = cha.lookupClass(TypeReference.findOrCreate(PythonTypes.pythonLoader, base));
          if (next != null) work.push(next);
        }
    }
    return ret;
  }

  /**
   * Builds the class hierarchy for the fixture.
   *
   * @return The class hierarchy.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private IClassHierarchy classHierarchy()
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    return builder.getClassHierarchy();
  }

  /**
   * Looks up a program class of the fixture.
   *
   * @param cha The class hierarchy.
   * @param name The program class's name in the fixture.
   * @return The class.
   */
  private IClass programClass(IClassHierarchy cha, String name) {
    IClass klass =
        cha.lookupClass(
            TypeReference.findOrCreate(
                PythonTypes.pythonLoader,
                TypeName.string2TypeName("Lscript " + FILE + "/" + name)));
    assertNotNull("The program class exists: " + name, klass);
    return klass;
  }
}
