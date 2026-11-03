package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A subscript adding an axis to an array a Keras dataset loader returns is a new NumPy array, which
 * the pointer analysis allocates in the program, as it does one of an {@code ndarray} (<a
 * href="https://github.com/wala/ML/issues/400">wala/ML#400</a>).
 */
public class TestKerasArraySubscript extends AbstractTensorTest {

  private static final String FILE = "tf2_test_keras_array_subscript.py";

  /** {@code x_train[..., None]}. */
  @Test
  public void testEllipsisNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_ellipsis_none",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UINT_8, 60000, 28, 28, 1))));
  }

  /** {@code x_train[:2]}. */
  @Test
  public void testSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_slice", 1, 1, Map.of(2, Set.of(TensorType.of(UINT_8, 2, 28, 28))));
  }

  /** {@code x_train[..., None].astype(np.float32) / 255.0}. */
  @Test
  public void testScaled() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE, "consume_scaled", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 60000, 28, 28, 1))));
  }

  /**
   * {@code x_train[:2, ..., None]}: a subscript with a slice lowers to the {@code slice} builtin,
   * so it is typed as one adding an axis to the slice.
   */
  @Test
  public void testSliceEllipsisNone() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_slice_ellipsis_none",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UINT_8, 2, 28, 28, 1))));
  }

  /**
   * A subscript adding an axis to an array read off a tuple, whose dtype the analysis does not
   * resolve: the result is a tensor of the subscript's shape and an unknown dtype, not no tensor.
   */
  @Test
  public void testUnknownDType() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unknown_dtype", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 3, 4, 1))));
  }

  /**
   * Each subscript adding an axis, and arithmetic over one, is one NumPy array allocated in the
   * script, not the loader's array: a result that aliased its receiver, which the loader's summary
   * allocates, would not pass. A slice of the loader's array still aliases it.
   */
  @Test
  public void testResultsAllocatedInTheProgram() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    for (String sink : new String[] {"consume_ellipsis_none", "consume_scaled"}) {
      int nodes = 0;
      for (CGNode node : builder.getCallGraph()) {
        if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
          continue;
        nodes++;
        PointerKey parameter =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
        Set<InstanceKey> keys = new HashSet<>();
        builder.getPointerAnalysis().getPointsToSet(parameter).forEach(keys::add);
        assertEquals(sink + "'s parameter: " + keys + ".", 1, keys.size());
        InstanceKey key = keys.iterator().next();
        assertEquals(
            sink + "'s object's kind.", NumpyTypes.NDARRAY_TYPE, key.concreteType().getReference());
        assertTrue(
            sink + "'s object is allocated outside the script: " + key + ".",
            key instanceof AllocationSiteInNode allocation
                && allocation
                    .getNode()
                    .getMethod()
                    .getDeclaringClass()
                    .getName()
                    .toString()
                    .endsWith(FILE));
      }
      assertEquals("Nodes of " + sink + ".", 1, nodes);
    }
  }

  /**
   * {@code x_train[:2, ..., None]} has a slice, so it takes the slice's allocation rule, under
   * which a subscript of a Keras dataset loader's array still aliases the loader's array
   * (wala/ML#841).
   */
  @Test
  public void testSliceEllipsisNoneAliasesTheLoadersArray() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    String sink = "consume_slice_ellipsis_none";
    int nodes = 0;
    for (CGNode node : builder.getCallGraph()) {
      if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) continue;
      nodes++;
      PointerKey parameter =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      Set<InstanceKey> keys = new HashSet<>();
      builder.getPointerAnalysis().getPointsToSet(parameter).forEach(keys::add);
      assertEquals(sink + "'s parameter: " + keys + ".", 1, keys.size());
      assertEquals(
          sink + "'s object's kind.",
          "Ltensorflow/keras/datasets/mnist/x_train",
          keys.iterator().next().concreteType().getName().toString());
    }
    assertEquals("Nodes of " + sink + ".", 1, nodes);
  }
}
