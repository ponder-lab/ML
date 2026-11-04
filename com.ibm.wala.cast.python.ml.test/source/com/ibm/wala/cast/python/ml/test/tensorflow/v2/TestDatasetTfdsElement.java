package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.ElementMirrorKey;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.AfterClass;
import org.junit.Test;

/**
 * The element of a dataset that TensorFlow Datasets' {@code tfds.load} returns under {@code
 * as_supervised=True} is an {@code (input, label)} pair (<a
 * href="https://github.com/wala/ML/issues/164">wala/ML#164</a>): the pair is no tensor, and each
 * member is a leaf of the element's copy, an object of its own, typed by the dataset's generator at
 * its position. Their types depend on the named dataset's features, so each is a tensor of unknown
 * shape and dtype. Without {@code as_supervised}, or with it false or not a literal, an element is
 * a dict of features the analysis cannot know, and the dataset keeps the generic element.
 */
public class TestDatasetTfdsElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_tfds_element.py";

  /** The builder whose points-to sets the leaf checks read, built once for the class. */
  private static volatile PythonSSAPropagationCallGraphBuilder builder;

  /** Releases the builder. */
  @AfterClass
  public static void releaseBuilder() {
    builder = null;
  }

  /**
   * The builder, whose call graph is built once for the class. The test methods run in parallel, so
   * one waits for another's build rather than reading its call graph while it is being built.
   *
   * @return The builder, with its call graph built.
   * @throws Exception On analysis error.
   */
  private PythonSSAPropagationCallGraphBuilder builder() throws Exception {
    synchronized (TestDatasetTfdsElement.class) {
      if (builder == null) {
        PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder built = engine.defaultCallGraphBuilder();
        built.makeCallGraph(built.getOptions());
        builder = built;
      }
      return builder;
    }
  }

  /**
   * The keys the parameter of the named sink points to, over every node of the sink.
   *
   * @param sink The sink's name.
   * @return The keys, or {@code null} when the sink has no node.
   * @throws Exception On analysis error.
   */
  private Set<InstanceKey> parameterKeys(String sink) throws Exception {
    PythonSSAPropagationCallGraphBuilder analyzed = builder();
    Set<InstanceKey> ret = null;
    for (CGNode node : analyzed.getCallGraph()) {
      if (!node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) continue;
      if (ret == null) ret = HashSetFactory.make();
      PointerKey parameter =
          analyzed.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      for (InstanceKey key : analyzed.getPointerAnalysis().getPointsToSet(parameter)) ret.add(key);
    }
    return ret;
  }

  /**
   * Checks that the named sink's parameter is a tensor of unknown shape and dtype, and that it
   * points to leaves of the element's copy alone.
   *
   * @param sink The sink's name.
   * @return The keys the parameter points to.
   * @throws Exception On analysis error.
   */
  private Set<InstanceKey> testLeaf(String sink) throws Exception {
    test(FILE, sink, 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
    Set<InstanceKey> keys = parameterKeys(sink);
    assertTrue(sink + " has a node.", keys != null);
    assertFalse(sink + "'s parameter points to an object.", keys.isEmpty());
    for (InstanceKey key : keys)
      assertTrue(
          sink + "'s parameter points to a leaf of the element's copy, not " + key + ".",
          builder().isElementMirrorLeaf(key));
    return keys;
  }

  /**
   * Checks that the named sink's parameter, a whole element, is a tensor of unknown shape and
   * dtype, and that it points to the generic element's allocations, not to a copy or its leaves.
   *
   * @param sink The sink's name.
   * @throws Exception On analysis error.
   */
  private void testGeneric(String sink) throws Exception {
    test(FILE, sink, 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
    Set<InstanceKey> keys = parameterKeys(sink);
    assertTrue(sink + " has a node.", keys != null);
    assertFalse(sink + "'s parameter points to an object.", keys.isEmpty());
    for (InstanceKey key : keys)
      assertFalse(
          sink + "'s parameter points to the generic element, not " + key + ".",
          key instanceof ElementMirrorKey);
  }

  /**
   * A supervised element is a tuple, no tensor.
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
   * The pair's input is a leaf of its own.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairX() throws Exception {
    testLeaf("consume_pair_x");
  }

  /**
   * The pair's label is a leaf of its own, distinct from the input.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testPairY() throws Exception {
    Set<InstanceKey> labels = testLeaf("consume_pair_y");
    Set<InstanceKey> inputs = parameterKeys("consume_pair_x");
    for (InstanceKey label : labels)
      assertFalse(
          "The label " + label + " is an object distinct from the input.", inputs.contains(label));
  }

  /**
   * A batch of the supervised pair batches each member: the input.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedX() throws Exception {
    testLeaf("consume_batched_x");
  }

  /**
   * The batched label.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testBatchedY() throws Exception {
    testLeaf("consume_batched_y");
  }

  /**
   * Without {@code as_supervised}, an element is a dict of features, and the dataset keeps the
   * generic element.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testOmitted() throws Exception {
    testGeneric("consume_omitted");
  }

  /**
   * With {@code as_supervised=False}, an element is a dict of features, and the dataset keeps the
   * generic element.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testFalse() throws Exception {
    testGeneric("consume_false");
  }

  /**
   * With an {@code as_supervised} that is no literal, an element is read as a dict of features, and
   * the dataset keeps the generic element.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testConditional() throws Exception {
    testGeneric("consume_conditional");
  }

  /**
   * Without call-string context, one {@code load} node stands for every call, so the supervised
   * calls and the others reach one dataset, and its element is not mirrored: the element reads as
   * the generic one its generator types, as before the pair was modeled.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testJoinedPair() throws ClassHierarchyException, CancelException, IOException {
    test(0, FILE, "consume_pair", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Without call-string context, the pair's input reads as the generic element's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testJoinedPairX() throws ClassHierarchyException, CancelException, IOException {
    test(0, FILE, "consume_pair_x", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * The supervised pair zipped with another dataset is a tuple, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZippedPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zipped_pair", 0, 0);
  }

  /**
   * The input of the supervised pair zipped with another dataset is a leaf of its own.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testZippedX() throws Exception {
    testLeaf("consume_zipped_x");
  }

  /**
   * The element of the dataset zipped with the supervised one keeps its own type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZippedZ() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zipped_z", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }
}
