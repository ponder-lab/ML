package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the analysis of chains of binary operators in which each operator's operands are the
 * results of the two before it: the types the chains' last values read, and that the analysis
 * classifies an operand's origins, and tests a value for being a scalar expression, a number of
 * times that does not grow with the number of paths of operators to it (<a
 * href="https://github.com/wala/ML/issues/1275">wala/ML#1275</a>). Classified at every visit, the
 * operands of a chain of thirty operators were classified tens of millions of times.
 */
public class TestOperandOriginChains extends AbstractTensorTest {

  /** Three chains of thirty operators: over tensors, over arrays, and over both. */
  private static final String CHAIN = "tf2_test_operand_origin_chain.py";

  /** Two chains of eighteen operators, each in a loop that carries its first two values. */
  private static final String LOOP = "tf2_test_operand_origin_loop.py";

  /** A chain of thirty operators over Python numbers, whose last value scales a tensor. */
  private static final String SCALAR_CHAIN = "tf2_test_scalar_expression_chain.py";

  /** The last value of the chain over tensors. */
  @Test
  public void testTensorChain() throws ClassHierarchyException, CancelException, IOException {
    test(CHAIN, "consume_tf", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The last value of the chain over arrays, and the last array of the chain over both. */
  @Test
  public void testArrayChain() throws ClassHierarchyException, CancelException, IOException {
    test(CHAIN, "consume_np", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The last value of the chain over both, a tensor. */
  @Test
  public void testMixedChain() throws ClassHierarchyException, CancelException, IOException {
    test(CHAIN, "consume_mixed", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The last value of the chain over tensors in a loop. */
  @Test
  public void testTensorLoop() throws ClassHierarchyException, CancelException, IOException {
    test(LOOP, "consume_tf", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The last value of the chain over arrays in a loop. */
  @Test
  public void testArrayLoop() throws ClassHierarchyException, CancelException, IOException {
    test(LOOP, "consume_np", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A tensor scaled by the last value of the chain over Python numbers keeps its type. */
  @Test
  public void testScalarChain() throws ClassHierarchyException, CancelException, IOException {
    test(SCALAR_CHAIN, "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The analysis classifies a few thousand operands for the three chains of thirty operators.
   * Classifying each again at every visit, it classified more than forty million.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testChainClassifications() throws Exception {
    Counts counts = this.count(CHAIN);
    assertTrue(counts.classifications() + " classifications", counts.classifications() < 10_000);
  }

  /**
   * As {@link #testChainClassifications()}, for the values tested for being a scalar expression:
   * tens of thousands, where testing a value again at every visit tested more than eighty million.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testChainScalarTests() throws Exception {
    Counts counts = this.count(CHAIN);
    assertTrue(counts.scalarTests() + " scalar tests", counts.scalarTests() < 100_000);
  }

  /**
   * As {@link #testChainClassifications()}, for the two chains of eighteen operators in loops,
   * whose operands reach themselves: under a thousand classifications, where the analysis
   * classified more than a hundred thousand.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testLoopClassifications() throws Exception {
    Counts counts = this.count(LOOP);
    assertTrue(counts.classifications() + " classifications", counts.classifications() < 5_000);
    assertTrue(counts.scalarTests() + " scalar tests", counts.scalarTests() < 25_000);
  }

  /**
   * The analysis tests about a hundred values of the chain over Python numbers for being a scalar
   * expression. Testing each again at every visit, it tested more than thirteen million.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testScalarChainScalarTests() throws Exception {
    Counts counts = this.count(SCALAR_CHAIN);
    assertTrue(counts.scalarTests() + " scalar tests", counts.scalarTests() < 1_000);
  }

  /**
   * What a fixture's analysis computed.
   *
   * @param classifications How many operand classifications it computed, rather than replayed.
   * @param scalarTests How many values it tested for being a scalar expression, rather than
   *     answered from a test's earlier answers.
   */
  private record Counts(long classifications, long scalarTests) {}

  /**
   * Analyzes a fixture and counts what the analysis computed.
   *
   * @param fixture The fixture.
   * @return The counts.
   */
  private Counts count(String fixture) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    return new Counts(
        OperandOriginWalk.computations(builder),
        ElementWiseOperation.scalarExpressionTests(builder));
  }
}
