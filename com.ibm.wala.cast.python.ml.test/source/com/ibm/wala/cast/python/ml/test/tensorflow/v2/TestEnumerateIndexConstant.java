package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * An {@code enumerate} index over a list or tuple literal of known length has the values {@code 0}
 * through {@code n - 1}, and {@code len} of one has the value {@code n}, so a size computed from
 * either folds, and a function the index is passed to receives those constants (<a
 * href="https://github.com/wala/ML/issues/1246">wala/ML#1246</a>). Once the list may grow, by
 * {@code append} as the analysis solves or by a route only the built call graph shows, such as
 * {@code extend}, either may be any {@code int}.
 */
public class TestEnumerateIndexConstant extends AbstractTensorTest {

  private static final String FILE = "tf2_test_enumerate_index_constant.py";

  /** A vector whose one extent is not known. */
  private static final TensorType TENSOR_UNRESOLVED_FLOAT32 =
      new TensorType(FLOAT_32, List.of(UnresolvedDim.INSTANCE));

  /** An index over a list a parameter holds, which every call passes a one-element literal. */
  @Test
  public void testFromParameter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_parameter", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** An index over a one-element literal. */
  @Test
  public void testFromLiteral() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_literal", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** The length of a three-element literal. */
  @Test
  public void testFromLen() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_len", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** The length of a three-element literal, returned by the function that takes it. */
  @Test
  public void testFromReturnedLen() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_returned_len", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** A constant, the control. */
  @Test
  public void testFromConstant() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_constant", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** An index over a literal that {@code append} grows, which may be any {@code int}. */
  @Test
  public void testFromAppended() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_appended", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /**
   * An index over a literal that {@code extend} grows, which only the scan of the built call graph
   * finds, so it may be any {@code int}.
   */
  @Test
  public void testFromExtended() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_extended", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /** The length of a literal that {@code append} grows, which may be any {@code int}. */
  @Test
  public void testFromAppendedLen() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_appended_len", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /**
   * The index {@code message} is passed points to exactly the constant {@code 0} in every context,
   * as a client reads a scalar parameter's points-to set to decide it has one value.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testIndexParameterIsConstant() throws Exception {
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    List<Object> values = new ArrayList<>();
    boolean found = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().contains(".message.")) continue;
      found = true;
      // `message(states, edge_type_idx)`: value number 1 is the function, 3 the index.
      PointerKey index = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 3);
      for (InstanceKey key : builder.getPointerAnalysis().getPointsToSet(index))
        values.add(key instanceof ConstantKey<?> constant ? constant.getValue() : key);
    }
    assertTrue("message is not in the call graph", found);
    assertEquals(List.of(0), values);
  }

  /**
   * An index over a one-element literal that {@code enumerate} starts at {@code 1}, which does not
   * count from {@code 0}, so it may be any {@code int}.
   */
  @Test
  public void testFromStarted() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_started", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /** The same, with {@code start} passed by keyword. */
  @Test
  public void testFromStartedKeyword()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_from_started_keyword", 1, 1, Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }

  /** The same, with {@code start} passed through a {@code **} argument. */
  @Test
  public void testFromStartedDoubleStarred()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_from_started_double_starred",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNRESOLVED_FLOAT32)));
  }
}
