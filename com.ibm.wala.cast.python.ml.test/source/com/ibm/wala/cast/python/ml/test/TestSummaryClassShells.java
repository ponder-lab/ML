package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.loader.PythonLoader;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;

/**
 * Tests for the summary-modeled class shells of <a
 * href="https://github.com/wala/ML/issues/118">wala/ML#118</a>: a source class subclassing a
 * framework base modeled only by an XML method summary (e.g. {@code tf.keras.layers.Layer} or
 * {@code tf.keras.Model}) resolves that base in the class hierarchy instead of falling back to
 * {@code object}.
 */
public class TestSummaryClassShells extends TestPythonMLCallGraphShape {

  /** The canonical {@code TypeName} string of the {@code tf.keras.layers.Layer} summary shell. */
  private static final String LAYER = "Ltensorflow/keras/layers/Layer";

  /**
   * The {@code TypeName} string of the {@code tensorflow/keras/Model} alias shell covering the
   * {@code tf.keras.Model} subclass spelling (<a
   * href="https://github.com/wala/ML/issues/662">wala/ML#662</a>).
   */
  private static final String MODEL_ALIAS = "Ltensorflow/keras/Model";

  /** The {@code TypeName} string of Python's {@code object}. */
  private static final String OBJECT = PythonTypes.object.getName().toString();

  /**
   * A subclass written against the aliased attribute path {@code tf.keras.layers.Layer} (under
   * {@code import tensorflow as tf}) has the {@code Layer} summary shell as its superclass.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built
   * @throws CancelException if the analysis is canceled
   * @throws IOException if the test file cannot be read
   */
  @Test
  public void testLayerSubclassSuperclass()
      throws ClassHierarchyException, CancelException, IOException {
    checkSuperclassChain("tf2_test_layer_subclass.py", "MyLayer", LAYER, OBJECT);
  }

  /**
   * A subclass written against the bare name {@code Layer} (under {@code from
   * tensorflow.keras.layers import Layer}) has the {@code Layer} summary shell as its superclass.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built
   * @throws CancelException if the analysis is canceled
   * @throws IOException if the test file cannot be read
   */
  @Test
  public void testLayerSubclassSuperclassFromImport()
      throws ClassHierarchyException, CancelException, IOException {
    checkSuperclassChain("tf2_test_layer_subclass2.py", "MyLayer", LAYER, OBJECT);
  }

  /**
   * A subclass written against the aliased attribute path {@code tf.keras.Model} has the {@code
   * tensorflow/keras/Model} alias shell as its superclass (<a
   * href="https://github.com/wala/ML/issues/662">wala/ML#662</a>).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built
   * @throws CancelException if the analysis is canceled
   * @throws IOException if the test file cannot be read
   */
  @Test
  public void testModelSubclassSuperclass()
      throws ClassHierarchyException, CancelException, IOException {
    checkSuperclassChain("tf2_test_model_subclass.py", "MyModel", MODEL_ALIAS, OBJECT);
  }

  /**
   * A subclass written against the bare name {@code Model} (under {@code from
   * tensorflow.keras.models import Model}) still falls back to {@code object}: the canonical {@code
   * tensorflow/keras/models/Model} instance type carries the {@code __call__}/{@code call} summary
   * bodies, and a method-less shell under that name would shadow the bypass-registered class
   * serving them (see the CAUTION in {@code tensorflow.xml}).
   *
   * <p>TODO: Expect the canonical type as the superclass once shells carry the summary's own
   * methods per <a href="https://github.com/wala/ML/issues/106">wala/ML#106</a>.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built
   * @throws CancelException if the analysis is canceled
   * @throws IOException if the test file cannot be read
   */
  @Test
  public void testModelSubclassSuperclassFromImport()
      throws ClassHierarchyException, CancelException, IOException {
    checkSuperclassChain("tf2_test_model_subclass2.py", "MyModel", OBJECT);
  }

  /**
   * Asserts that {@code className} in {@code filename} has exactly the given superclass chain above
   * it, and that each shell on the chain engages {@link PythonLoader.PythonClass}-gated machinery.
   *
   * @param filename the test file defining {@code className}
   * @param className the source class whose chain to check
   * @param expectedChain the expected superclass {@code TypeName} strings, from the immediate
   *     superclass upward (ending at {@code Lobject})
   * @throws ClassHierarchyException if the class hierarchy cannot be built
   * @throws CancelException if the analysis is canceled
   * @throws IOException if the test file cannot be read
   */
  private void checkSuperclassChain(String filename, String className, String... expectedChain)
      throws ClassHierarchyException, CancelException, IOException {
    PythonAnalysisEngine<TensorTypeAnalysis> engine = makeEngine(filename);
    engine.defaultCallGraphBuilder();
    IClassHierarchy cha = engine.getClassHierarchy();

    IClass current =
        cha.lookupClass(
            TypeReference.findOrCreate(
                PythonTypes.pythonLoader, "Lscript " + filename + "/" + className));
    assertNotNull("expected the source subclass in the hierarchy", current);

    for (String expected : expectedChain) {
      IClass superclass = current.getSuperclass();
      assertNotNull("expected a superclass named " + expected, superclass);
      assertEquals(expected, superclass.getReference().getName().toString());

      if (!expected.equals(OBJECT)) {
        // The shell engages `IPythonClass`-gated machinery.
        assertTrue(
            "expected the " + expected + " shell to be a PythonLoader class",
            superclass instanceof PythonLoader.PythonClass);
      }

      current = superclass;
    }
  }
}
