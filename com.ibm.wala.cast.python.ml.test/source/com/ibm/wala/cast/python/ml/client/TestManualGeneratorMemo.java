package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import java.io.File;
import java.util.Collections;
import org.junit.Test;

/**
 * Tests that a manual generator is dispatched once per (node, type) while an analysis resolves, not
 * again at every step of a dataset's size read (<a
 * href="https://github.com/wala/ML/issues/1271">wala/ML#1271</a>).
 */
public class TestManualGeneratorMemo extends AbstractTensorTest {

  /** A chain of dataset transformations, batched twice, whose element several consumers read. */
  private static final String FILE = "tf2_test_dataset_nested_leaf_reads.py";

  /**
   * The analysis dispatches no more manual generators than the call graph has nodes, the most
   * (node, type) pairs it can ask for with no {@code Sequential} composition in the program, and
   * releases them when it ends. Unmemoized, this fixture's 47 nodes were dispatched 108 times.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testDispatchedOncePerNode() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    long nodes = builder.getCallGraph().getNumberOfNodes();
    engine.performAnalysis(builder);
    long dispatches = TensorGenerator.manualDispatchCount(builder);
    assertTrue("The analysis dispatches a manual generator.", dispatches > 0);
    assertTrue(
        "The analysis dispatched " + dispatches + " manual generators for " + nodes + " nodes.",
        dispatches <= nodes);
    assertFalse(
        "The analysis clears its memoized manual generators.",
        TensorGenerator.hasManualGenerators(builder));
  }
}
