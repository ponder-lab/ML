package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.client.ShapeOperandRank.Rank.MAY_BE_RANK_0;
import static com.ibm.wala.cast.python.ml.client.ShapeOperandRank.Rank.RANK_1;
import static com.ibm.wala.cast.python.ml.client.ShapeOperandRank.Rank.UNKNOWN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.ShapeOperandRank;
import com.ibm.wala.cast.python.ml.client.ShapeOperandRank.Rank;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ssa.SSAInstruction;
import java.io.File;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests {@link ShapeOperandRank#ofReshape}, the client-facing authority on the rank of a {@code
 * tf.reshape} call's shape operand (<a
 * href="https://github.com/wala/ML/issues/1147">wala/ML#1147</a>), on {@code
 * tf2_test_reshape_shape_rank.py}. Each function of the fixture makes one reshape, and every
 * reshape whose operand a test reads as {@link Rank#MAY_BE_RANK_0} raises under {@code tf.function}
 * on TensorFlow 2.9.3, while every one it reads as {@link Rank#RANK_1} traces.
 */
public class TestShapeOperandRank extends TestPythonMLCallGraphShape {

  private static PythonSSAPropagationCallGraphBuilder builder;

  private static TensorTypeAnalysis analysis;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        new TestShapeOperandRank()
            .makeEngine(Collections.<File>emptyList(), "tf2_test_reshape_shape_rank.py");
    builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    analysis = engine.performAnalysis(builder);
  }

  /**
   * The ranks of the shape operands of the reshapes in a fixture function, across every node of the
   * function.
   *
   * @param function The function's name.
   * @return The ranks.
   */
  private static Set<Rank> ranksIn(String function) {
    Set<Rank> ret = EnumSet.noneOf(Rank.class);
    for (CGNode node : builder.getCallGraph()) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      if (!CallTargets.isProgramCode(node)
          || !name.endsWith("/" + function)
          || node.getIR() == null) continue;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof PythonInvokeInstruction call
            && TensorFlowApis.of(builder, node, call).contains("tensorflow.reshape"))
          ret.add(ShapeOperandRank.ofReshape(builder, analysis, node, call));
    }
    assertTrue("`" + function + "` must reshape in the call graph.", !ret.isEmpty());
    return ret;
  }

  /** A number literal, {@code tf.reshape(x, -1)}. */
  @Test
  public void testLiteral() {
    assertEquals(Set.of(MAY_BE_RANK_0), ranksIn("by_literal"));
  }

  /** A number literal by keyword, {@code tf.reshape(x, shape=-1)}. */
  @Test
  public void testKeyword() {
    assertEquals(Set.of(MAY_BE_RANK_0), ranksIn("by_keyword"));
  }

  /** A rank-0 tensor, {@code tf.reshape(x, tf.constant(-1))}. */
  @Test
  public void testScalarTensor() {
    assertEquals(Set.of(MAY_BE_RANK_0), ranksIn("by_scalar_tensor"));
  }

  /** A NumPy scalar, {@code tf.reshape(x, np.int32(-1))}. */
  @Test
  public void testNumpyScalar() {
    assertEquals(Set.of(MAY_BE_RANK_0), ranksIn("by_numpy_scalar"));
  }

  /** A list of numbers, {@code tf.reshape(x, [-1])}. */
  @Test
  public void testList() {
    assertEquals(Set.of(RANK_1), ranksIn("by_list"));
  }

  /** A tuple of numbers, {@code tf.reshape(x, (3, 4))}. */
  @Test
  public void testTuple() {
    assertEquals(Set.of(RANK_1), ranksIn("by_tuple"));
  }

  /**
   * A list whose element is a computed int, {@code [x.shape[0] * x.shape[1]]}: the element has no
   * evidence, and is read as a scalar, since a non-scalar element would make the reshape fail
   * eagerly as well as in a graph.
   */
  @Test
  public void testListOfComputed() {
    assertEquals(Set.of(RANK_1), ranksIn("by_list_of_computed"));
  }

  /** A rank-1 tensor, {@code tf.reshape(x, tf.shape(y))}. */
  @Test
  public void testShapeTensor() {
    assertEquals(Set.of(RANK_1), ranksIn("by_shape_tensor"));
  }

  /** A NumPy vector, {@code tf.reshape(x, np.array([-1]))}. */
  @Test
  public void testNumpyVector() {
    assertEquals(Set.of(RANK_1), ranksIn("by_numpy_vector"));
  }

  /**
   * A computed int, {@code tf.reshape(x, x.shape[0] * x.shape[1])}, has no instance key
   * (wala/ML#805), so its rank is unknown, never rank 1. With wala/ML#805, it may be rank 0.
   */
  @Test
  public void testComputedInt() {
    assertEquals(Set.of(UNKNOWN), ranksIn("by_computed_int"));
  }

  /** A number on one path and a list on another: some value is rank 0. */
  @Test
  public void testEither() {
    assertEquals(Set.of(MAY_BE_RANK_0), ranksIn("by_either"));
  }

  /**
   * A list on one path and a value the query cannot classify on another, here an instance of a
   * program class with no tensor type: the list alone does not make the operand rank 1.
   */
  @Test
  public void testListOrUnclassified() {
    assertEquals(Set.of(UNKNOWN), ranksIn("by_list_or_unclassified"));
  }

  /**
   * A starred argument, {@code tf.reshape(*leading, y, -1)} with {@code leading} empty: the slot
   * after the tensor's holds the vector {@code y}, while the shape is the scalar {@code -1}, so the
   * operand cannot be aligned.
   */
  @Test
  public void testStarred() {
    assertEquals(Set.of(UNKNOWN), ranksIn("by_starred"));
  }

  /** A list on one path and a list with a list element on another, {@code [[12]]}. */
  @Test
  public void testListOrNested() {
    assertEquals(Set.of(UNKNOWN), ranksIn("by_list_or_nested"));
  }

  /**
   * A list on one path and a list with a rank-1 tensor element on another, {@code
   * [tf.constant([12])]}.
   */
  @Test
  public void testListOrVectorElement() {
    assertEquals(Set.of(UNKNOWN), ranksIn("by_list_or_vector_element"));
  }
}
