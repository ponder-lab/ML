package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_LAYER_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_MODEL_TYPE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeReference;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.junit.Test;

/**
 * Tests {@link Util#allocatedClass} on {@code tf2_test_allocated_class.py} (<a
 * href="https://github.com/wala/ML/issues/1165">wala/ML#1165</a>), where each sink receives one
 * value, and checks the classes it returns against {@link
 * com.ibm.wala.cast.python.ml.types.TensorFlowTypes#KERAS_LAYER_TYPE} and {@link
 * com.ibm.wala.cast.python.ml.types.TensorFlowTypes#KERAS_MODEL_TYPE} with the class hierarchy.
 */
public class TestAllocatedClass extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_allocated_class.py";

  private static PythonSSAPropagationCallGraphBuilder builder;

  /**
   * Builds the fixture's call graph, once.
   *
   * @return Its builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestAllocatedClass().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * The allocated classes of the values a sink receives, one per instance key.
   *
   * @param sink The sink's name.
   * @return The classes {@link Util#allocatedClass} gives its parameter's instance keys.
   */
  private static List<Optional<IClass>> allocatedClasses(String sink) {
    List<Optional<IClass>> ret = new ArrayList<>();
    int nodes = 0;
    for (CGNode node : builder().getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) {
        nodes++;
        PointerKey parameter =
            builder().getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
        for (InstanceKey key : builder().getPointerAnalysis().getPointsToSet(parameter))
          ret.add(Util.allocatedClass(key));
      }
    assertEquals("Expecting one node for " + sink + ".", 1, nodes);
    assertEquals("Expecting one value for " + sink + ".", 1, ret.size());
    return ret;
  }

  /**
   * The one class a sink's value was allocated by, which must be known.
   *
   * @param sink The sink's name.
   * @return The class.
   */
  private static IClass allocatedClass(String sink) {
    Optional<IClass> ret = allocatedClasses(sink).get(0);
    assertTrue("The class of " + sink + "'s value is known.", ret.isPresent());
    return ret.get();
  }

  /**
   * Whether a class extends a modeled class.
   *
   * @param type The class.
   * @param base The modeled class.
   * @return Whether {@code type} is {@code base} or a subclass of it.
   */
  private static boolean isSubclassOf(IClass type, TypeReference base) {
    IClassHierarchy cha = builder().getClassHierarchy();
    IClass baseClass = cha.lookupClass(base);
    assertNotNull("The class " + base + " is modeled.", baseClass);
    return cha.isSubclassOf(type, baseClass);
  }

  /**
   * Asserts a sink's value is allocated by a class of the program with the given name.
   *
   * @param sink The sink's name.
   * @param name The class's name, after the script's.
   * @return The class.
   */
  private static IClass assertProgramClass(String sink, String name) {
    IClass ret = allocatedClass(sink);
    assertEquals("Lscript " + FILE + "/" + name, ret.getName().toString());
    return ret;
  }

  /** A Keras layer subclass held in a model's field is an instance of that subclass, a layer. */
  @Test
  public void testLayerSubclass() {
    IClass type = assertProgramClass("consume_encoder", "Encoder");
    assertTrue(isSubclassOf(type, KERAS_LAYER_TYPE));
    assertFalse(isSubclassOf(type, KERAS_MODEL_TYPE));
  }

  /** A Keras model subclass is an instance of that subclass, a model. */
  @Test
  public void testModelSubclass() {
    IClass type = assertProgramClass("consume_holder", "Holder");
    assertTrue(isSubclassOf(type, KERAS_MODEL_TYPE));
    assertFalse(isSubclassOf(type, KERAS_LAYER_TYPE));
  }

  /** An object of a plain class is an instance of that class, neither a layer nor a model. */
  @Test
  public void testPlainClass() {
    IClass type = assertProgramClass("consume_plain", "Plain");
    assertFalse(isSubclassOf(type, KERAS_LAYER_TYPE));
    assertFalse(isSubclassOf(type, KERAS_MODEL_TYPE));
  }

  /** A subclass of a layer subclass is an instance of the most derived class, still a layer. */
  @Test
  public void testIndirectLayerSubclass() {
    IClass type = assertProgramClass("consume_derived", "Derived");
    assertTrue(isSubclassOf(type, KERAS_LAYER_TYPE));
  }

  /** A built-in layer a summary allocates is an instance of its modeled class, a layer. */
  @Test
  public void testBuiltinLayer() {
    IClass type = allocatedClass("consume_dense");
    assertEquals("Ltensorflow/keras/layers/Dense", type.getName().toString());
    assertTrue(isSubclassOf(type, KERAS_LAYER_TYPE));
    assertFalse(isSubclassOf(type, KERAS_MODEL_TYPE));
  }

  /** A constant has no allocating class. */
  @Test
  public void testConstant() {
    assertFalse(allocatedClasses("consume_constant").get(0).isPresent());
  }

  /** An {@code int} of unknown value, the engine's type-level key, has no allocating class. */
  @Test
  public void testScalarOfUnknownValue() {
    assertFalse(allocatedClasses("consume_length").get(0).isPresent());
  }

  /** A class object, which the program builds by its definition, is no instance of its class. */
  @Test
  public void testClassObject() {
    assertFalse(allocatedClasses("consume_class").get(0).isPresent());
  }

  /** A summary's callable, such as {@code tf.keras.layers.Dense} itself, is no layer instance. */
  @Test
  public void testLibraryCallable() {
    assertFalse(allocatedClasses("consume_callable").get(0).isPresent());
  }
}
