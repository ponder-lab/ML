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

  /**
   * An unreached {@code counts.update} on a dict display built in the same function leaves a list
   * elsewhere complete.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testDisplayReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds_display.py").isComplete());
  }

  /**
   * An unreached {@code update} on a dict built by the {@code dict} builtin, read through a {@code
   * global} declaration, leaves a list elsewhere complete.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testGlobalBuiltinReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds_global_builtin.py").isComplete());
  }

  /**
   * An unreached {@code update} on a dict built by the {@code dict} builtin leaves a list elsewhere
   * complete when only a variable of another function's frame binds {@code dict} to a function of
   * the program's own.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testFrameLocalBuiltinReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds_local_builtin.py").isComplete());
  }

  /**
   * Unreached {@code tf.keras.backend.update} calls, on TensorFlow imported by the enclosing frame,
   * leave a list elsewhere complete.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testEnclosingLibraryReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds_enclosing.py").isComplete());
  }

  /**
   * An unreached {@code defaults().update(source)}, where {@code defaults} is the module's own
   * function returning a shared dict, may add to that dict.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testNonBuiltinCallReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_call.py").isComplete());
  }

  /**
   * An unreached {@code factory().extend(items)}, where {@code factory} is a variable of the
   * enclosing frame, may add to any list.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testClosureCallReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_closure.py").isComplete());
  }

  /**
   * An unreached {@code make().extend(items)}, where {@code make} is a parameter, may add to any
   * list.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testParameterCallReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_factory.py").isComplete());
  }

  /**
   * An unreached {@code tables[key].update(entries)} may add to any dict.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testSubscriptReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_subscript.py").isComplete());
  }

  /**
   * An unreached {@code CACHE.update(entries)} on a global the function declares may add to the
   * dict that global is written.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testGlobalReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_global.py").isComplete());
  }

  /**
   * An unreached {@code os.environ.update(settings)} may add to any dict.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testUnlistedAttributeReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_environ.py").isComplete());
  }

  /**
   * An unreached {@code tf.backend.update}, which is not {@code tf.keras.backend.update}, may add
   * to any dict.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test
  public void testUnlistedLibraryReceiver() throws Exception {
    assertFalse(consumed("tf2_test_unreached_adds_library_unlisted.py").isComplete());
  }

  /**
   * An unreached {@code counts.update} on a dict built by the {@code dict} builtin leaves a list
   * elsewhere complete, though another function defines a {@code dict} of its own inside it.
   *
   * <p>TODO: The nested definition makes the builtin look rebound everywhere until
   * https://github.com/wala/ML/issues/1196 is fixed.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test(expected = AssertionError.class)
  public void testNestedBuiltinNameReceiver() throws Exception {
    assertTrue(consumed("tf2_test_unreached_adds_nested_builtin.py").isComplete());
  }
}
