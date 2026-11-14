package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
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
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a class's decorators are applied as Python applies them, and that its keywords are
 * recorded on it (<a href="https://github.com/wala/ML/issues/1260">wala/ML#1260</a>). The front end
 * once dropped both, so a class's name was bound to the class itself whatever its decorators
 * returned. Each class's method passes its argument to its own consumer.
 */
public class TestClassDecorators extends AbstractTensorTest {

  private static final String FILE = "tf2_test_class_decorators.py";

  /** A decorator that registers the class and returns it: the class is constructed. */
  @Test
  public void testRegistered() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_registered", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A decorator that returns another class: the name is bound to the other class, so a call of the
   * name constructs it.
   */
  @Test
  public void testReplaced() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_replacement", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /**
   * A decorator that returns another class, with the class constructed in a function body: the
   * body's read of the module-level name sees what the decorator returned, as a module-level read
   * does.
   */
  @Test
  public void testReplacedInFunction()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_swapped", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 10))));
  }

  /**
   * Two decorators: the inner one, which returns another class, applies first, and the outer one,
   * which returns what it is given, second, so the name is bound to the inner one's class.
   */
  @Test
  public void testStacked() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stacked", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A bare {@code @dataclass}, which the {@code dataclasses} summary returns the class from, so the
   * class keeps its methods.
   */
  @Test
  public void testDataclass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dataclass", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * Two classes under the same bare summary decorator keep their own class objects: only {@code
   * Point} is constructed, so {@code Unused}'s method is not reached. The decorator's applications
   * once shared one node, which bound each name to both classes.
   */
  @Test(expected = AssertionError.class)
  public void testBareDecoratorKeepsClassesApart()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unused", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * {@code tf.keras.utils.register_keras_serializable(...)}, which its summary returns the class
   * from, so the class keeps its methods.
   */
  @Test
  public void testSerializable() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_serializable", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6))));
  }

  /** A class with a {@code metaclass=} keyword and no decorator is constructed as before. */
  @Test
  public void testMetaclassClassReached()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_abstract", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 7))));
  }

  /**
   * A decorator that returns a wrapper function: the name is bound to the wrapper, which constructs
   * the class. A zero-argument {@code super()} in the class's own initializer should read the
   * class, as Python's {@code __class__} cell does, so the base's initializer is reached. The front
   * end reads it through the class's name, which holds the wrapper, so the base is not found.
   *
   * <p>TODO: wala/ML#1266; once {@code super()} reads the class itself, flip to a plain
   * {@code @Test} reading {@code (8,)} float32.
   */
  @Test(expected = AssertionError.class)
  public void testSuperReadsTheClass()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_super", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 8))));
  }

  /**
   * A decorator neither the program nor a summary defines resolves no callee, so the class's name
   * is bound to nothing and the class is not constructed. That is a modeling gap in the decorator's
   * library, filled by a summary for it as {@code dataclass} has, not by the front end.
   */
  @Test(expected = AssertionError.class)
  public void testUnmodeledDecoratorUnbinds()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unmodeled", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 9))));
  }

  /**
   * The class statement's keywords are recorded on the class, each with its value's dotted name,
   * and a class that passes none records none.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testKeywordsRecorded() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    IClassHierarchy cha = builder.getClassHierarchy();
    assertEquals(Map.of("metaclass", "abc.ABCMeta"), keywords(cha, "Abstract"));
    assertTrue(keywords(cha, "Registered").isEmpty());
  }

  /**
   * The keywords a class of the fixture records.
   *
   * @param cha The class hierarchy.
   * @param name The class's name.
   * @return Its recorded keywords.
   */
  private static Map<String, String> keywords(IClassHierarchy cha, String name) {
    IClass cls =
        cha.lookupClass(
            TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lscript " + FILE + "/" + name));
    assertNotNull(name + " is defined", cls);
    return ((IPythonClass) cls).getKeywords();
  }
}
