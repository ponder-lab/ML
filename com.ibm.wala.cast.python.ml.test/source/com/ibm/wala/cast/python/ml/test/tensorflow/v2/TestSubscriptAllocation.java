package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The result of subscripting an array or a tensor with an index tuple holding {@code ...}, {@code
 * None} ({@code tf.newaxis}) or a slice is a new array or tensor, which the pointer analysis
 * allocates (<a href="https://github.com/wala/ML/issues/400">wala/ML#400</a>).
 */
public class TestSubscriptAllocation extends AbstractTensorTest {

  private static final String FILE = "tf2_test_subscript_allocation.py";

  /** {@code a[..., tf.newaxis]} on an array. */
  @Test
  public void testEllipsisNewaxis() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_ellipsis_newaxis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2, 1))));
  }

  /** {@code a[..., None]} on an array. */
  @Test
  public void testEllipsisNone() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_ellipsis_none", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2, 1))));
  }

  /** {@code a[None, ...]} on an array. */
  @Test
  public void testNoneEllipsis() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_none_ellipsis", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 3, 2))));
  }

  /** {@code a[:2, ..., tf.newaxis]} on an array. */
  @Test
  public void testSliceEllipsisNewaxis()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_slice_ellipsis_newaxis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2, 1))));
  }

  /** {@code a[:, 0, None]} on an array. */
  @Test
  public void testSliceNone() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_slice_none", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 1))));
  }

  /** {@code t[..., tf.newaxis]} on a tensor. */
  @Test
  public void testTensorEllipsisNewaxis()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_tensor_ellipsis_newaxis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2, 1))));
  }

  /** {@code t[:2, ..., tf.newaxis]} on a tensor. */
  @Test
  public void testTensorSliceEllipsisNewaxis()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_tensor_slice_ellipsis_newaxis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2, 1))));
  }

  /** Every form's result points to an object. */
  @Test
  public void testResultsPointToObjects() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    List<String> empty = new ArrayList<>();
    for (String sink :
        new String[] {
          "consume_ellipsis_newaxis",
          "consume_ellipsis_none",
          "consume_none_ellipsis",
          "consume_slice_ellipsis_newaxis",
          "consume_slice_none",
          "consume_tensor_ellipsis_newaxis",
          "consume_tensor_slice_ellipsis_newaxis"
        }) {
      boolean found = false;
      for (CGNode node : builder.getCallGraph()) {
        if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
          continue;
        found = true;
        PointerKey parameter =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
        if (builder.getPointerAnalysis().getPointsToSet(parameter).isEmpty()) empty.add(sink);
      }
      assertTrue("No node for " + sink + ".", found);
    }
    assertTrue("Parameters pointing to nothing: " + empty + ".", empty.isEmpty());
  }
}
