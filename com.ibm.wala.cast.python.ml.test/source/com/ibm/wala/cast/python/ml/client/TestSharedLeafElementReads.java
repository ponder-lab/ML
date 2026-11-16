package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeFalse;

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
 * Tests that the values holding a mirrored dataset element leaf share the read of the leaf's
 * element, so the element's datasets are walked once per leaf and not once per value holding it (<a
 * href="https://github.com/wala/ML/issues/1273">wala/ML#1273</a>).
 */
public class TestSharedLeafElementReads extends AbstractTensorTest {

  /**
   * A dataset of a dictionary of eight members, sixteen of whose reads pass through two shared
   * helpers, so each helper's result holds all eight members.
   */
  private static final String FILE = "tf2_test_dataset_leaf_values.py";

  /** Each value passed on is a batch of a member, which all have one type. */
  @Test
  public void testConsume() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 4, 3))));
  }

  /**
   * The analysis computes a leaf's element a few times, as the queries the computation reads
   * settle, not once per value holding the leaf for each read of the value. Unshared, this
   * fixture's elements were computed 514 times; shared, 9 times, for its 18 leaves.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testElementComputedPerLeaf() throws Exception {
    // The check mode computes a kept value again at every read of it, to compare the two.
    assumeFalse(Boolean.getBoolean(WorklistTypeResolver.CHECK_TRACKED_PROPERTY));
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    long leaves = 0;
    for (InstanceKey key : builder.getPointerAnalysis().getInstanceKeys())
      if (key instanceof ElementMirrorKey mirror && mirror.isLeaf()) leaves++;
    assertTrue("The fixture has mirrored leaves.", leaves > 0);
    engine.performAnalysis(builder);
    long computations = DatasetTupleElementGenerator.readComputations(builder);
    assertTrue("The analysis reads a leaf's element.", computations > 0);
    assertTrue(
        "The analysis computed " + computations + " element reads for " + leaves + " leaves.",
        computations <= 2 * leaves);
  }
}
