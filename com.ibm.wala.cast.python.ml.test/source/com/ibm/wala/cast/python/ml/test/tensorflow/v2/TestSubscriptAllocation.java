package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The result of subscripting an array or a tensor with an index tuple holding {@code ...}, {@code
 * None} ({@code tf.newaxis}) or a slice, with {@code None} alone, or with a list is a new array or
 * tensor, which the pointer analysis allocates at the subscript; so is a full integer index's
 * scalar, modeled as a 0-d array (<a href="https://github.com/wala/ML/issues/400">wala/ML#400</a>).
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

  /**
   * {@code a[None]} on an array.
   *
   * <p>TODO: Typed as one row, {@code (2,)}, until https://github.com/wala/ML/issues/1192 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testNoneAlone() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_none_alone", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 3, 2))));
  }

  /**
   * {@code a[0, 1]} on an array: a scalar of the array's dtype.
   *
   * <p>TODO: Typed as one row, {@code (2,)}, until https://github.com/wala/ML/issues/1192 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testIntegerPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_integer_pair", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /**
   * {@code a[[0, 2]]} on an array: the rows a list selects.
   *
   * <p>TODO: The leading axis reads as unresolved until https://github.com/wala/ML/issues/798 is
   * fixed.
   */
  @Test(expected = AssertionError.class)
  public void testFancy() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fancy", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
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

  /**
   * {@code t[0, 1]} on a tensor: a rank-0 tensor.
   *
   * <p>TODO: Typed as one row, {@code (2,)}, until https://github.com/wala/ML/issues/1192 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testTensorIntegerPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tensor_integer_pair", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /** {@code a[..., None].astype(np.float32) / 255.0}: the subscript's result, cast, then scaled. */
  @Test
  public void testCastScaled() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_cast_scaled", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2, 1))));
  }

  /** {@code tf.cast(t[..., tf.newaxis], tf.float32) / 255.0} on a tensor. */
  @Test
  public void testTensorCastScaled() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_tensor_cast_scaled",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2, 1))));
  }

  /**
   * Each subscript's result is an object of the receiver's kind that the subscript itself
   * allocates: in the script's node, at the subscript's read or at the {@code slice} call a slice
   * lowers to. A result that aliased its receiver, which a library summary allocates, would not
   * pass, nor would one of another kind.
   */
  @Test
  public void testResultsAllocatedAtTheSubscript() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);

    Map<String, TypeReference> kinds = new LinkedHashMap<>();
    for (String sink :
        new String[] {
          "consume_ellipsis_newaxis",
          "consume_ellipsis_none",
          "consume_none_ellipsis",
          "consume_slice_ellipsis_newaxis",
          "consume_slice_none",
          "consume_none_alone",
          "consume_integer_pair",
          "consume_fancy"
        }) kinds.put(sink, NumpyTypes.NDARRAY_TYPE);
    for (String sink :
        new String[] {
          "consume_tensor_ellipsis_newaxis",
          "consume_tensor_slice_ellipsis_newaxis",
          "consume_tensor_integer_pair"
        }) kinds.put(sink, TensorFlowTypes.TENSOR_TYPE);

    for (Map.Entry<String, TypeReference> entry : kinds.entrySet()) {
      String sink = entry.getKey();
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
            sink + "'s object's kind.", entry.getValue(), key.concreteType().getReference());
        assertTrue(
            sink + "'s object is no allocation: " + key + ".", key instanceof AllocationSiteInNode);
        AllocationSiteInNode allocation = (AllocationSiteInNode) key;
        CGNode allocator = allocation.getNode();
        assertTrue(
            sink + "'s object is allocated outside the script: " + allocator + ".",
            allocator.getMethod().getDeclaringClass().getName().toString().endsWith(FILE));
        SSAInstruction at = null;
        for (SSAInstruction instruction : allocator.getIR().getInstructions())
          if (instruction != null
              && instruction.iIndex() == allocation.getSite().getProgramCounter()) at = instruction;
        assertTrue(
            sink + "'s object is allocated at " + at + ", not a subscript.",
            at instanceof AstPropertyRead || at instanceof SSAAbstractInvokeInstruction);
      }
      assertEquals("Nodes of " + sink + ".", 1, nodes);
    }
  }
}
