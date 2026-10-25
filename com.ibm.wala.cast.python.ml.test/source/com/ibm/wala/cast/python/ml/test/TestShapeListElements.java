package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Tests that the elements of a tensor's static shape list are Python ints to a points-to client, on
 * {@code tf2_test_shape_list_elements.py} (<a
 * href="https://github.com/wala/ML/issues/1104">wala/ML#1104</a>): each sink receives a list, and
 * the test reads whether {@link ContainerElements} finds a Python {@code int}'s type-level key
 * among its elements.
 */
public class TestShapeListElements extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_shape_list_elements.py";

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
            new TestShapeListElements().makeEngine(Collections.<File>emptyList(), FILE);
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
   * Whether a Python {@code int}'s type-level key is among the elements of the list a sink
   * receives.
   *
   * @param sink The sink's name.
   * @return {@code true} iff some element may be a Python {@code int} of unknown value.
   */
  private static boolean holdsPythonInt(String sink) {
    List<PointerKey> parameters = new ArrayList<>();
    for (CGNode node : builder().getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
        parameters.add(
            builder().getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for " + sink + ".", 1, parameters.size());
    for (PointerKey element : ContainerElements.ofValue(builder(), parameters.get(0)).allElements())
      for (InstanceKey ik : builder().getPointerAnalysis().getPointsToSet(element))
        if (Util.isPythonScalarKey(ik)) return true;
    return false;
  }

  /** The list {@code t.shape.as_list()} holds Python ints. */
  @Test
  public void testAsList() {
    assertTrue(holdsPythonInt("consume_as_list"));
  }

  /**
   * A {@code shape_list} utility's list, {@code [dynamic[i] if s is None else s for i, s in
   * enumerate(static)]}, holds the Python ints of the static shape it keeps.
   */
  @Test
  public void testShapeList() {
    assertTrue(holdsPythonInt("consume_shape_list"));
  }

  /** A list of {@code tf.shape} elements alone holds no Python int. */
  @Test
  public void testTensorList() {
    assertFalse(holdsPythonInt("consume_tensor_list"));
  }
}
