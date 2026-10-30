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
 * Tests which calls in a function the call graph does not reach may add an element to any
 * container. A call that adds to a container the function builds itself, or a TensorFlow function
 * that resizes no container, does not; a call on a parameter may.
 */
public class TestUnreachedAdds extends TestPythonMLCallGraphShape {

  /**
   * The elements of the value a fixture's {@code consume} receives.
   *
   * @param file The fixture.
   * @return The parameter's elements.
   * @throws Exception if the fixture cannot be analyzed.
   */
  private ContainerElements consumed(String file) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    List<PointerKey> keys = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/consume"))
        keys.add(builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for consume.", 1, keys.size());
    return ContainerElements.ofValue(builder, keys.get(0));
  }

  /**
   * An unreached {@code words.add} on a set built in the same function leaves a list elsewhere
   * complete.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testFreshReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds.py").isComplete());
  }

  /**
   * An unreached {@code tf.keras.backend.update} leaves a list elsewhere complete.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testLibraryReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds_library.py").isComplete());
  }

  /**
   * An unreached {@code dict().update(source)}, where {@code dict} is the module's own function
   * returning a shared dict, may add to that dict.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testShadowedBuiltinReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_shadowed.py").isComplete());
  }

  /**
   * An unreached {@code target.update(source)} on a parameter may add to any dict.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testParameterReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_parameter.py").isComplete());
  }
}
