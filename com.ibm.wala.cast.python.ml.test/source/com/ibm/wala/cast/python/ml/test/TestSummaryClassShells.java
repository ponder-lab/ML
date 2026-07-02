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
 * framework base modeled only by an XML method summary (e.g. {@code tf.keras.layers.Layer})
 * resolves that base in the class hierarchy instead of falling back to {@code object}.
 */
public class TestSummaryClassShells extends TestPythonMLCallGraphShape {

  /** The canonical {@link TypeReference} of the {@code tf.keras.layers.Layer} summary shell. */
  private static final TypeReference LAYER =
      TypeReference.findOrCreate(PythonTypes.pythonLoader, "Ltensorflow/keras/layers/Layer");

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
    checkLayerSubclass("tf2_test_layer_subclass.py");
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
    checkLayerSubclass("tf2_test_layer_subclass2.py");
  }

  /**
   * Asserts that {@code MyLayer} in {@code filename} has the {@code Layer} summary shell as its
   * superclass, that the shell engages {@link PythonLoader.PythonClass}-gated machinery, and that
   * the shell itself is positioned at {@code object}.
   *
   * @param filename the test file defining {@code MyLayer}
   * @throws ClassHierarchyException if the class hierarchy cannot be built
   * @throws CancelException if the analysis is canceled
   * @throws IOException if the test file cannot be read
   */
  private void checkLayerSubclass(String filename)
      throws ClassHierarchyException, CancelException, IOException {
    PythonAnalysisEngine<TensorTypeAnalysis> engine = makeEngine(filename);
    engine.defaultCallGraphBuilder();
    IClassHierarchy cha = engine.getClassHierarchy();

    IClass subclass =
        cha.lookupClass(
            TypeReference.findOrCreate(
                PythonTypes.pythonLoader, "Lscript " + filename + "/MyLayer"));
    assertNotNull("expected the source subclass in the hierarchy", subclass);

    IClass superclass = subclass.getSuperclass();
    assertNotNull("expected the Layer summary shell as the superclass", superclass);
    assertEquals(LAYER, superclass.getReference());

    // The shell engages `IPythonClass`-gated machinery.
    assertTrue(
        "expected the shell to be a PythonLoader class",
        superclass instanceof PythonLoader.PythonClass);

    // The shell itself is positioned at `object`, per its `super` declaration in tensorflow.xml.
    IClass shellSuper = superclass.getSuperclass();
    assertNotNull("expected the shell to link to object", shellSuper);
    assertEquals(PythonTypes.object, shellSuper.getReference());
  }
}
