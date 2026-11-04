package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.AfterClass;
import org.junit.Test;

/**
 * The element of {@code tf.data.Dataset.from_generator} has the structure the dataset declares, its
 * {@code output_signature} or else its {@code output_types} (<a
 * href="https://github.com/wala/ML/issues/164">wala/ML#164</a>): a tuple or dict element is no
 * tensor, and each member keeps its own shape and dtype. Each member is a leaf of the element's
 * copy, an object of its own, rather than a generic element allocation or nothing.
 */
public class TestDatasetFromGeneratorElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_from_generator_element.py";

  /** The builder whose points-to sets the leaf checks read, built once for the class. */
  private static PythonSSAPropagationCallGraphBuilder builder;

  /** Releases the builder. */
  @AfterClass
  public static synchronized void releaseBuilder() {
    builder = null;
  }

  /**
   * The class's builder, built by the first test method that asks for it. The test methods run in
   * parallel, so the builder is published only once its call graph is complete.
   *
   * @return The builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestDatasetFromGeneratorElement().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * The keys the parameter of the named sink points to, over every node of the sink.
   *
   * @param sink The sink's name.
   * @return The keys, or {@code null} when the sink has no node.
   * @throws Exception On analysis error.
   */
  private Set<InstanceKey> parameterKeys(String sink) throws Exception {
    PythonSSAPropagationCallGraphBuilder builder = builder();
    Set<InstanceKey> ret = null;
    for (CGNode node : builder.getCallGraph()) {
      if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) continue;
      if (ret == null) ret = HashSetFactory.make();
      PointerKey parameter =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      for (InstanceKey key : builder.getPointerAnalysis().getPointsToSet(parameter)) ret.add(key);
    }
    return ret;
  }

  /**
   * Checks the named sink's parameter types, and that the parameter points to leaves of the
   * element's copy alone.
   *
   * @param sink The sink's name.
   * @param types The parameter's expected types.
   * @throws Exception On analysis error.
   */
  private void testLeaf(String sink, Set<TensorType> types) throws Exception {
    test(FILE, sink, 1, 1, Map.of(2, types));
    Set<InstanceKey> keys = parameterKeys(sink);
    assertTrue(sink + " has a node.", keys != null);
    assertFalse(sink + "'s parameter points to an object.", keys.isEmpty());
    for (InstanceKey key : keys)
      assertTrue(
          sink + "'s parameter points to a leaf of the element's copy, not " + key + ".",
          builder().isElementMirrorLeaf(key));
  }

  /**
   * Checks the named sink's parameter type, and that the parameter points to leaves of the
   * element's copy alone.
   *
   * @param sink The sink's name.
   * @param type The parameter's one expected type.
   * @throws Exception On analysis error.
   */
  private void testLeaf(String sink, TensorType type) throws Exception {
    testLeaf(sink, Set.of(type));
  }

  /**
   * A pair declared by {@code output_signature} is a tuple, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair", 0, 0);
  }

  /**
   * The pair's first member keeps its spec's type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairX() throws Exception {
    testLeaf("consume_pair_x", TENSOR_4_FLOAT32);
  }

  /**
   * The pair's second member keeps its own spec's type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairY() throws Exception {
    testLeaf("consume_pair_y", TENSOR_2_3_INT32);
  }

  /**
   * A dict declared by {@code output_signature}: the member read by its first key.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testKeyedX() throws Exception {
    testLeaf("consume_keyed_x", TensorType.of(FLOAT_32, 5));
  }

  /**
   * The member read by the dict's second key.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testKeyedY() throws Exception {
    testLeaf("consume_keyed_y", SCALAR_TENSOR_OF_INT64);
  }

  /**
   * A nested tuple: the first member of the inner pair.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testNestedX() throws Exception {
    testLeaf("consume_nested_x", TensorType.of(FLOAT_32, 6));
  }

  /**
   * The second member of the inner pair.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testNestedMask() throws Exception {
    testLeaf("consume_nested_mask", TensorType.of(FLOAT_32, 2));
  }

  /**
   * The outer pair's second member.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testNestedY() throws Exception {
    testLeaf("consume_nested_y", SCALAR_TENSOR_OF_INT32);
  }

  /**
   * A pair declared by the legacy {@code output_types}, shaped by {@code output_shapes}: the first
   * member.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testTypedX() throws Exception {
    testLeaf("consume_typed_x", TENSOR_4_FLOAT32);
  }

  /**
   * The second member of that pair.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testTypedY() throws Exception {
    testLeaf("consume_typed_y", TENSOR_2_3_INT32);
  }

  /**
   * A pair declared by {@code output_types} whose members share one dtype: the first member keeps
   * its own shape.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testSameDTypeX() throws Exception {
    testLeaf("consume_same_x", TENSOR_4_FLOAT32);
  }

  /**
   * The second member of that pair keeps its own shape.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testSameDTypeY() throws Exception {
    testLeaf("consume_same_y", TensorType.of(FLOAT_32, 2, 3));
  }

  /**
   * A batch of the declared pair batches each member: the first. A generator's length is not known,
   * so a partial batch of unknown size is read beside the full one, as the dataset's static shape
   * has it.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedX() throws Exception {
    testLeaf(
        "consume_batched_x",
        Set.of(
            TensorType.of(FLOAT_32, 2, 4),
            new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(4)))));
  }

  /**
   * The second batched member, beside its partial batch of unknown size.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedY() throws Exception {
    testLeaf(
        "consume_batched_y",
        Set.of(
            TensorType.of(INT_32, 2, 2, 3),
            new TensorType(
                INT_32, List.of(new SymbolicDim("?"), new NumericDim(2), new NumericDim(3)))));
  }

  /**
   * A dataset whose call declares no structure, which the run time rejects, keeps the generic
   * element its generator types: the first member, as unknown.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testUndeclaredX() throws Exception {
    test(FILE, "consume_undeclared_x", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
    Set<InstanceKey> keys = parameterKeys("consume_undeclared_x");
    assertTrue("consume_undeclared_x has a node.", keys != null);
    assertFalse("consume_undeclared_x's parameter points to an object.", keys.isEmpty());
    for (InstanceKey key : keys)
      assertFalse(
          "consume_undeclared_x's parameter points to the generic element, not " + key + ".",
          builder().isElementMirrorLeaf(key));
  }
}
