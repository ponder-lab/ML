package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
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
 * The element of {@code Dataset.enumerate()} is a tuple of an int64 scalar index and the receiver's
 * element, copied as the receiver's is (<a
 * href="https://github.com/wala/ML/issues/1190">wala/ML#1190</a>): the index is an int64 scalar,
 * and each member of the receiver's element keeps its own shape and dtype. Each member is a leaf of
 * the element's copy, an object of its own, rather than a generic element allocation or nothing.
 */
public class TestDatasetEnumerateElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_enumerate_element.py";

  /** The builder whose points-to sets the leaf checks read, built once for the class. */
  private static PythonSSAPropagationCallGraphBuilder builder;

  /** Releases the builder. */
  @AfterClass
  public static void releaseBuilder() {
    builder = null;
  }

  /**
   * The keys the parameter of the named sink points to, over every node of the sink.
   *
   * @param sink The sink's name.
   * @return The keys, or {@code null} when the sink has no node.
   * @throws Exception On analysis error.
   */
  private Set<InstanceKey> parameterKeys(String sink) throws Exception {
    if (builder == null) {
      PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
      builder = engine.defaultCallGraphBuilder();
      builder.makeCallGraph(builder.getOptions());
    }
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
   * @param type The parameter's one expected type.
   * @throws Exception On analysis error.
   */
  private void testLeaf(String sink, TensorType type) throws Exception {
    test(FILE, sink, 1, 1, Map.of(2, Set.of(type)));
    Set<InstanceKey> keys = parameterKeys(sink);
    assertTrue(sink + " has a node.", keys != null);
    assertFalse(sink + "'s parameter points to an object.", keys.isEmpty());
    for (InstanceKey key : keys)
      assertTrue(
          sink + "'s parameter points to a leaf of the element's copy, not " + key + ".",
          builder.isElementMirrorLeaf(key));
  }

  /**
   * The index of an enumerated flat dataset.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testFlatIndex() throws Exception {
    testLeaf("consume_flat_index", SCALAR_TENSOR_OF_INT64);
  }

  /**
   * The element of an enumerated flat dataset.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testFlatElement() throws Exception {
    testLeaf("consume_flat_x", TensorType.of(FLOAT_32, 3));
  }

  /**
   * The index of an enumerated dataset whose element is a pair.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairIndex() throws Exception {
    testLeaf("consume_pair_index", SCALAR_TENSOR_OF_INT64);
  }

  /**
   * The pair's first member keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairFirst() throws Exception {
    testLeaf("consume_pair_x", TensorType.of(FLOAT_32, 4));
  }

  /**
   * The pair's second member keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairSecond() throws Exception {
    testLeaf("consume_pair_y", TensorType.of(INT_32, 2));
  }

  /**
   * The pair itself, the enumerated element's second member, is a tuple, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPairElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_element", 0, 0, Map.of());
  }

  /**
   * The index and the pair's two members are three objects, none shared with another.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairMembersSeparate() throws Exception {
    Set<InstanceKey> index = parameterKeys("consume_pair_index");
    Set<InstanceKey> x = parameterKeys("consume_pair_x");
    Set<InstanceKey> y = parameterKeys("consume_pair_y");
    for (Set<InstanceKey> keys : List.of(index, x, y))
      assertTrue("Each member points to an object.", keys != null && !keys.isEmpty());
    assertTrue("The index is not the first member.", Collections.disjoint(index, x));
    assertTrue("The index is not the second member.", Collections.disjoint(index, y));
    assertTrue("The members are distinct.", Collections.disjoint(x, y));
  }

  /**
   * A batch of an enumerated pair dataset batches the index.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedIndex() throws Exception {
    testLeaf("consume_batched_index", TensorType.of(INT_64, 2));
  }

  /**
   * A batch of an enumerated pair dataset batches the pair's first member.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedFirst() throws Exception {
    testLeaf("consume_batched_x", TensorType.of(FLOAT_32, 2, 4));
  }

  /**
   * A batch of an enumerated pair dataset batches the pair's second member.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedSecond() throws Exception {
    testLeaf("consume_batched_y", TensorType.of(INT_32, 2, 2));
  }

  /**
   * The index of an enumerated dataset whose element is a dict.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testDictIndex() throws Exception {
    testLeaf("consume_dict_index", SCALAR_TENSOR_OF_INT64);
  }

  /**
   * A dict member read by its key keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testDictFirst() throws Exception {
    testLeaf("consume_dict_a", TensorType.of(FLOAT_32, 3));
  }

  /**
   * Another dict member read by its key keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testDictSecond() throws Exception {
    testLeaf("consume_dict_b", SCALAR_TENSOR_OF_INT32);
  }

  /**
   * A member of an enumerated {@code from_generator} element, whose structure the generator's
   * signature declares, is its own object.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testGeneratedX() throws Exception {
    testLeaf("consume_generated_x", TensorType.of(FLOAT_32, 3));
  }

  /**
   * The index of an enumerated zip.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedIndex() throws Exception {
    testLeaf("consume_zipped_index", SCALAR_TENSOR_OF_INT64);
  }

  /**
   * A member of an enumerated zip's element, the element of the first zipped dataset, keeps its own
   * type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedX() throws Exception {
    testLeaf("consume_zipped_x", TENSOR_4_FLOAT32);
  }

  /**
   * Another member of an enumerated zip's element keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedY() throws Exception {
    testLeaf("consume_zipped_y", TENSOR_2_INT32);
  }

  /**
   * The index of an enumerated dataset zipped with another.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedInnerIndex() throws Exception {
    testLeaf("consume_zipped_inner_index", SCALAR_TENSOR_OF_INT64);
  }

  /**
   * The element of an enumerated dataset zipped with another keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedInnerX() throws Exception {
    testLeaf("consume_zipped_inner_x", TENSOR_4_FLOAT32);
  }

  /**
   * The element of the dataset zipped with an enumerated one keeps its own type.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedInnerY() throws Exception {
    testLeaf("consume_zipped_inner_y", TENSOR_2_INT32);
  }
}
