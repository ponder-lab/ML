package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.stream.StreamSupport;
import org.junit.Test;

/**
 * Tests that the function {@code tf.map_fn} is given is called with an element of {@code elems},
 * {@code elems} without its leading dimension, so its parameter is typed from the map (<a
 * href="https://github.com/wala/ML/issues/942">wala/ML#942</a>). Before, the summary returned
 * {@code elems} without calling the function, and a function called only through {@code map_fn} had
 * no call-graph node.
 */
public class TestMapFnCallback extends AbstractTensorTest {

  private static final String FILE = "tf2_test_map_fn_callback.py";

  /**
   * A function called only by {@code map_fn}, over rows of four {@code float32} values cast from
   * {@code int32}. Its locals are the parameter, the stacked result, and the two scalar elements
   * {@code bbox[0]} and {@code bbox[2]} that an integer subscript of the parameter reads.
   */
  @Test
  public void testCallbackOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "unpack", 1, 4, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * A function called by {@code map_fn} over rows of three {@code int32} values and directly with
   * three {@code float32} values holds both types, and no row of another {@code map_fn} call in the
   * same scope.
   */
  @Test
  public void testCallbackAndDirectCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "scale", 1, 2, Map.of(2, Set.of(TENSOR_3_INT32, TensorType.of(FLOAT_32, 3))));
  }

  /** A function another {@code map_fn} call in the same scope maps over rows of one value. */
  @Test
  public void testSecondCallback()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "double", 1, 2, Map.of(2, Set.of(TensorType.of(FLOAT_64, 1))));
  }

  /**
   * Given a tuple of tensors as {@code elems}, {@code map_fn} passes its function a tuple of rows,
   * which is not a tensor.
   */
  @Test
  public void testTupleElems()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "add_pair", 0, 0);
  }

  /** A function stored in a variable and never called has no call-graph node. */
  @Test
  public void testStoredNeverCalled() throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), new String[] {FILE});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    Util.addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertTrue("The mapped function is reached.", reaches(CG, "scale"));
    assertFalse("The stored function is not reached.", reaches(CG, "unused"));
  }

  /**
   * Whether a function of the fixture has a call-graph node.
   *
   * @param CG The call graph.
   * @param function The function's name.
   * @return {@code true} iff some node's method is the function's body.
   */
  private static boolean reaches(CallGraph CG, String function) {
    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    return StreamSupport.stream(CG.spliterator(), false)
        .map(CGNode::getMethod)
        .anyMatch(m -> m.getSignature().equals(signature));
  }
}
