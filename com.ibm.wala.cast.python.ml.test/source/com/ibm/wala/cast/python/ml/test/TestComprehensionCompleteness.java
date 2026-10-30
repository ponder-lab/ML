package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Tests that a list a comprehension builds has all its elements known, on {@code
 * tf2_test_comprehension_complete.py}. The comprehension's body writes each element under a key the
 * analysis cannot name, and also under the append contents, so the write under the unnamed key adds
 * no element the analysis does not see.
 */
public class TestComprehensionCompleteness extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_comprehension_complete.py";

  private static PythonSSAPropagationCallGraphBuilder builder;

  /**
   * Analyzes the fixture, once.
   *
   * @return Its builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestComprehensionCompleteness().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        engine.performAnalysis(b);
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * The elements of the value a sink receives.
   *
   * @param sink The sink's name.
   * @return Its parameter's elements.
   */
  private static ContainerElements ofValue(String sink) {
    List<PointerKey> keys = new ArrayList<>();
    for (CGNode node : builder().getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
        keys.add(builder().getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for " + sink + ".", 1, keys.size());
    return ContainerElements.ofValue(builder(), keys.get(0));
  }

  /** A comprehension over {@code range(3)} builds a list whose elements are all known. */
  @Test
  public void testRange() {
    assertTrue(ofValue("consume_range").isComplete());
  }

  /** So does a comprehension of a constant tuple. */
  @Test
  public void testConstant() {
    assertTrue(ofValue("consume_constant").isComplete());
  }

  /** A list written under a parameter key may hold an element the analysis does not see. */
  @Test
  public void testKeyedWrite() {
    assertFalse(ofValue("consume_keyed").isComplete());
  }
}
