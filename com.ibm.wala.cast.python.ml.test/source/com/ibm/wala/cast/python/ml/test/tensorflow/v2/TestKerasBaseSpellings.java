package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a program class whose base is a Keras layer or model is one in the class hierarchy
 * however the base is spelled (wala/ML#1127): through the standalone {@code keras} package,
 * TensorFlow's legacy {@code tensorflow.python.keras}, or {@code tf.compat}, or as a built-in layer
 * that the summaries do not model. Before, each such base had no summary class, so the program
 * class's superclass was {@code object} and its declared bases reached no Keras class.
 */
public class TestKerasBaseSpellings extends AbstractTensorTest {

  private static final String FILE = "tf2_test_keras_base_spellings.py";

  /** The Keras layer base class. */
  private static final String LAYER = "Ltensorflow/keras/layers/Layer";

  /** The Keras model base class. */
  private static final String MODEL = "Ltensorflow/keras/models/Model";

  /** Bases spelled through the standalone {@code keras} package. */
  @Test
  public void testStandaloneKeras() throws ClassHierarchyException, CancelException, IOException {
    IClassHierarchy cha = this.classHierarchy();
    assertReaches(cha, "KerasLayer", LAYER);
    assertReaches(cha, "KerasModel", MODEL);
    assertReaches(cha, "KerasDense", "Ltensorflow/keras/layers/Dense");
    assertReaches(cha, "KerasDense", LAYER);
  }

  /**
   * Bases spelled through TensorFlow's legacy {@code tensorflow.python.keras}. Its classes are
   * distinct from {@code tf.keras}'s at run time but share their {@code __call__} contract.
   */
  @Test
  public void testLegacyKeras() throws ClassHierarchyException, CancelException, IOException {
    IClassHierarchy cha = this.classHierarchy();
    assertReaches(cha, "LegacyLayerSub", LAYER);
    assertReaches(cha, "BaseLayerSub", LAYER);
    assertReaches(cha, "TrainingModelSub", MODEL);
  }

  /** Bases spelled through {@code tf.compat}. */
  @Test
  public void testCompat() throws ClassHierarchyException, CancelException, IOException {
    IClassHierarchy cha = this.classHierarchy();
    assertReaches(cha, "CompatV1Layer", LAYER);
    assertReaches(cha, "CompatV1Model", MODEL);
    assertReaches(cha, "CompatV2Layer", LAYER);
  }

  /**
   * Built-in layers the summaries do not model. {@code MaxPooling2D} is an alias of the modeled
   * {@code MaxPool2D}, so it reaches {@code Layer} through it.
   */
  @Test
  public void testBuiltinLayers() throws ClassHierarchyException, CancelException, IOException {
    IClassHierarchy cha = this.classHierarchy();
    assertReaches(cha, "MaxPooling2DSub", "Ltensorflow/keras/layers/MaxPool2D");
    assertReaches(cha, "MaxPooling2DSub", LAYER);
    assertReaches(cha, "Conv3DSub", LAYER);
    assertReaches(cha, "MultiHeadAttentionSub", LAYER);
    assertReaches(cha, "MaxPool1DSub", "Ltensorflow/keras/layers/MaxPooling1D");
    assertReaches(cha, "MaxPool1DSub", LAYER);
  }

  /**
   * A subclass of {@code keras.layers.Dense} that does not override {@code call}: applying an
   * instance runs the {@code call} it inherits from {@code Dense}, so the result is a dense
   * layer's. Before, the superclass was {@code object}, so the subclass had no {@code call} and the
   * application resolved to nothing.
   */
  @Test
  public void testStandaloneKerasInheritedCall()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dense", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A subclass of the legacy {@code Layer} that creates its weight with the {@code add_weight} it
   * inherits from {@code Layer}, so the weight is a tensor and the product of the input and the
   * weight has the weight's last dimension. Before, the superclass was {@code object}, and the
   * weight was nothing.
   */
  @Test
  public void testLegacyKerasInheritedAddWeight()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_weighted", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }

  /** A program class named {@code Layer}, and its subclass, are not Keras layers. */
  @Test
  public void testProgramClassNamedLayer()
      throws ClassHierarchyException, CancelException, IOException {
    IClassHierarchy cha = this.classHierarchy();
    for (String name : List.of("Layer", "NotKeras")) {
      Set<String> bases = declaredBaseClosure(cha, name);
      assertFalse(name + "'s bases reach no Keras layer: " + bases, bases.contains(LAYER));
      assertFalse(name + "'s bases reach no Keras model: " + bases, bases.contains(MODEL));
    }
  }

  /**
   * Asserts that the declared bases of a program class of the fixture reach a class.
   *
   * @param cha The class hierarchy.
   * @param name The program class's name in the fixture.
   * @param expected The name of the class its declared bases reach.
   */
  private static void assertReaches(IClassHierarchy cha, String name, String expected) {
    Set<String> bases = declaredBaseClosure(cha, name);
    assertTrue(name + "'s bases reach " + expected + ": " + bases, bases.contains(expected));
  }

  /**
   * The names of the declared bases of a program class of the fixture, and of their declared bases
   * in turn, each looked up in the class hierarchy: the walk a client makes over {@link
   * IPythonClass#getBaseTypeNames()} to recognize a Keras layer. A class that records no bases
   * answers with its superclass.
   *
   * @param cha The class hierarchy.
   * @param name The program class's name in the fixture.
   * @return The base names reached.
   */
  private static Set<String> declaredBaseClosure(IClassHierarchy cha, String name) {
    IClass start =
        cha.lookupClass(
            TypeReference.findOrCreate(
                PythonTypes.pythonLoader,
                TypeName.string2TypeName("Lscript " + FILE + "/" + name)));
    assertNotNull("The program class exists: " + name, start);
    Set<String> ret = new LinkedHashSet<>();
    Deque<IClass> work = new ArrayDeque<>(List.of(start));
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
    return engine.defaultCallGraphBuilder().getClassHierarchy();
  }
}
