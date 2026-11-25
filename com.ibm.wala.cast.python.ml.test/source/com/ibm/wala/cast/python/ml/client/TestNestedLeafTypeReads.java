package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.ElementMirrorKey;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a value read of a mirrored dataset element leaf, whose reading reads upstream leaves,
 * reads each leaf's shapes and dtypes as queries of the resolver rather than directly (<a
 * href="https://github.com/wala/ML/issues/1270">wala/ML#1270</a>). Read directly, each nested leaf
 * was read again on both axes for every value, and a cycle of leaf readings recursed without bound.
 */
public class TestNestedLeafTypeReads extends AbstractTensorTest {

  /** A chain of transformations whose functions return the element's members, one per map. */
  private static final String FILE = "tf2_test_dataset_nested_leaf_reads.py";

  /** As {@link #FILE}, with one function shared by two maps. */
  private static final String SHARED = "tf2_test_dataset_shared_leaf_reads.py";

  /** The features, passed through every transformation of the chain, keep their type. */
  @Test
  public void testFeatures() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 2, 3))));
  }

  /** As {@link #testFeatures()}, for the labels. */
  @Test
  public void testLabels() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_label", 1, 1, Map.of(2, Set.of(TENSOR_1_2_INT32)));
  }

  /**
   * The analysis reads each leaf's types a few times, as its two queries are evaluated and
   * re-evaluated when a query they read changes, not once per value and per nesting. Read directly,
   * this fixture's 9 leaves were read 119 times.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testLeafTypesReadAsQueries() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    long leaves = 0;
    for (InstanceKey key : builder.getPointerAnalysis().getInstanceKeys())
      if (key instanceof ElementMirrorKey mirror && mirror.isLeaf()) leaves++;
    assertTrue("The fixture has mirrored leaves.", leaves > 0);
    engine.performAnalysis(builder);
    long reads = TensorGeneratorFactory.leafTypeReads(builder);
    assertTrue("The analysis reads a leaf's types.", reads > 0);
    assertTrue(
        "The analysis read " + reads + " leaf types for " + leaves + " leaves.",
        reads <= 3 * leaves);
  }

  /**
   * One function shared by two maps returns, at each, the leaves both maps hand it, so the leaves
   * of each map's element read the other's: their readings form a cycle, which the resolver
   * converges. Read directly, the cycle recursed until the stack overflowed. The features read as
   * the join of the element at both maps, through the batch between them, and each turn of the
   * cycle adds an axis, so the growth widening reads the unknown shape; at run time they are {@code
   * (1, 2, 3)}.
   *
   * <p>TODO: The features read {@code (1, 2, 3)} float32 alone once
   * https://github.com/wala/ML/issues/1305 is fixed.
   */
  @Test
  public void testSharedFunctionFeatures()
      throws ClassHierarchyException, CancelException, IOException {
    test(SHARED, "consume", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * As {@link #testSharedFunctionFeatures()}, for the labels, {@code (1, 2)} at run time.
   *
   * <p>TODO: The labels read {@code (1, 2)} int32 alone once https://github.com/wala/ML/issues/1305
   * is fixed.
   */
  @Test
  public void testSharedFunctionLabels()
      throws ClassHierarchyException, CancelException, IOException {
    test(SHARED, "consume_label", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_INT32)));
  }
}
