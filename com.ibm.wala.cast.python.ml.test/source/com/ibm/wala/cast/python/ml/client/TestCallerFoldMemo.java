package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.Pair;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a caller's guards, folded for its callees' argument reads, are memoized per builder
 * only when the fold read no query of the resolver (<a
 * href="https://github.com/wala/ML/issues/863">wala/ML#863</a>).
 */
public class TestCallerFoldMemo extends AbstractTensorTest {

  private static final String FILE = "tf2_test_caller_fold_memo.py";

  /** Both callers pass the consumer the tensor, whose guards both hold. */
  @Test
  public void testConsumer() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * While an analysis resolves, the caller whose guard folds from its own constants is memoized,
   * the caller whose guard folds from its argument's rank, which the fold reads from the resolver,
   * is not, and asking again gives the same callers.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testOnlyQueryFreeFoldsMemoized() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    List<CGNode> consumers = nodes(builder, "consume");
    CGNode constantCaller = node(builder, "constant_caller");
    CGNode rankCaller = node(builder, "rank_caller");
    WorklistTypeResolver.install(builder);
    try {
      List<Pair<CGNode, SSAAbstractInvokeInstruction>> first = callerInvokes(builder, consumers);
      assertEquals("Both callers' calls run.", 2, first.size());
      assertTrue(
          "The constant guard's fold is memoized.",
          TensorGenerator.hasCallerFold(builder, constantCaller));
      assertFalse(
          "The rank guard's fold is not memoized.",
          TensorGenerator.hasCallerFold(builder, rankCaller));
      long folds = TensorGenerator.callerFoldCount(builder);
      assertEquals("Asking again gives the same calls.", first, callerInvokes(builder, consumers));
      assertEquals(
          "Asking again folds only the rank guard's caller.",
          folds + 1,
          TensorGenerator.callerFoldCount(builder));
    } finally {
      WorklistTypeResolver.uninstall(builder, false);
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * The memoized folds are released with the other per-builder memos when the analysis ends.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAnalysisReleasesCallerFolds() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    assertTrue("The analysis folds a caller.", TensorGenerator.callerFoldCount(builder) > 0);
    assertFalse(
        "The analysis clears its memoized caller folds.", TensorGenerator.hasCallerFolds(builder));
  }

  /**
   * The calling invocations of each of some nodes, in order.
   *
   * @param builder The builder.
   * @param nodes The nodes.
   * @return Their calling invocations.
   */
  private static List<Pair<CGNode, SSAAbstractInvokeInstruction>> callerInvokes(
      PythonSSAPropagationCallGraphBuilder builder, List<CGNode> nodes) {
    List<Pair<CGNode, SSAAbstractInvokeInstruction>> ret = new ArrayList<>();
    for (CGNode node : nodes) ret.addAll(TensorGenerator.getCallerInvokes(builder, node));
    return ret;
  }

  /**
   * The fixture's nodes for a function, one per context.
   *
   * @param builder The builder.
   * @param function The function's name.
   * @return Its nodes.
   */
  private static List<CGNode> nodes(PythonSSAPropagationCallGraphBuilder builder, String function) {
    List<CGNode> ret = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + function))
        ret.add(node);
    assertFalse("Expecting a node for " + function + ".", ret.isEmpty());
    return ret;
  }

  /**
   * The fixture's node for a function.
   *
   * @param builder The builder.
   * @param function The function's name.
   * @return Its one node.
   */
  private static CGNode node(PythonSSAPropagationCallGraphBuilder builder, String function) {
    CGNode ret = null;
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + function)) {
        assertTrue("Expecting one node for " + function + ".", ret == null);
        ret = node;
      }
    assertNotNull("Expecting a node for " + function + ".", ret);
    return ret;
  }
}
