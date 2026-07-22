package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_0_0_9_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_2_27_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_4_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_784_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

import com.ibm.wala.cast.python.ml.client.BroadcastTo;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of unary math, linear-algebra, and shape-query op inference, carved from the {@link
 * TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestMathOps extends AbstractTensorTest {

  /**
   * Pins {@code top_p_logits(logits, p)}'s parameter type. Function body mirrors {@code
   * akanyaani/gpt-2-tensorflow2.0/sample.py}'s {@code top_p_logits}. Exercises several ops
   * currently routed through {@code ReadDataFallback} per wala/ML#449 ({@code tf.sort}, {@code
   * tf.cumsum}, {@code tf.stack}, {@code tf.range}, {@code tf.gather_nd}, {@code tf.where}), but
   * the parameter type of {@code logits} comes from its caller (a {@code tf.constant} with shape
   * {@code (1, 5)} dtype {@code float32}), so this test isolates caller-side propagation rather
   * than the body's op precision.
   *
   * <p>Empirically, {@code logits} is inferred as {@code (1, 5) float32} — concrete on both axes.
   * The caller's {@code tf.constant([[1.0, ..., 5.0]], dtype=tf.float32)} flows in cleanly, showing
   * that none of the body's {@code ReadDataFallback}-routed ops block caller-side propagation for
   * this function; the parameter type is fully resolved at the call site.
   */
  @Test
  public void testTopPLogits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_top_p_logits.py", "top_p_logits", 1, 12, Map.of(2, Set.of(TENSOR_1_5_FLOAT32)));
  }

  /**
   * Pins {@code _take_long_axis(arr, indices)}'s parameter types. Function body mirrors {@code
   * _take_long_axis} from {@code
   * LongmaoTeamTf/deep_recommenders/keras/models/retrieval/factorized_top_k.py}.
   *
   * <p>This fixture surfaced a real Ariadne bug: {@code tf.reshape(arr, tf.shape(other))} crashes
   * the analysis with {@code IllegalStateException} at {@code
   * TensorGenerator.getShapesFromShapeArgument} because the shape argument is a Tensor (the result
   * of {@code tf.shape(...)}) rather than a list/tuple literal. The {@code Reshape} generator
   * should gracefully degrade to ⊤ shape per the lattice conventions instead of throwing. Resolved
   * by <a href="https://github.com/wala/ML/issues/538">wala/ML#538</a>; parameter types pin
   * precisely.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/543">wala/ML#543</a>): the post-fix
   * local-tensor count of 9 captures every intermediate runtime-shape tensor flowing through the
   * body. Tighten once the body-level imprecision is addressed.
   */
  @Test
  public void testTakeAlongAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_take_along_axis.py",
        "_take_long_axis",
        2,
        10,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_2_INT32)));
  }

  /**
   * A negative k-CFA depth is invalid (the depth is the call-string length for the targeted context
   * selector) and is rejected at construction (wala/ML#379).
   */
  @Test(expected = IllegalArgumentException.class)
  public void testNegativeTargetedCfaDepthRejected() {
    new PythonTensorAnalysisEngine(
        emptyList(), PythonTensorAnalysisEngine.TENSORFLOW, /* targetedCfaDepth= */ -1);
  }

  @Test
  public void testSigmoid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testAbs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_abs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testAcos()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_acos.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testExp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_exp.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testExp()} exercising the keyword-argument call site {@code
   * tf.math.exp(x=...)}. {@link com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator}
   * routes argument resolution through {@code getArgumentPointsToSet(builder, paramPos, paramName)}
   * which resolves keyword args via {@code paramName}; without a kwarg fixture that branch is
   * dead-on-arrival in the test data.
   */
  @Test
  public void testExpKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_exp_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testTanh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tanh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRsqrt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rsqrt.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Companion to {@link #testRsqrt()} exercising the keyword-argument call site. */
  @Test
  public void testRsqrtKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rsqrt_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testLogSoftmax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log_softmax.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testLogSoftmax()} exercising the keyword-argument call site {@code
   * tf.nn.log_softmax(logits=...)}.
   */
  @Test
  public void testLogSoftmaxKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log_softmax_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testL2Normalize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_l2_normalize.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testSigmoid2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testRank()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rank.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_size.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Verifies that {@code tf.broadcast_to(x, [2, 3])} routes through the dedicated {@link
   * BroadcastTo} generator and emits shape {@code (2, 3)} (read from the {@code shape} argument's
   * literal list) with {@code float32} dtype (derived from the {@code input} tensor).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBroadcastTo()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_broadcast_to.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.broadcast_to(x, tf.shape(y))} &mdash; the runtime-tensor
   * shape-arg pattern. {@link com.ibm.wala.cast.python.ml.client.TensorGenerator
   * #getShapesFromShapeArgument} throws {@link IllegalStateException} for the runtime {@code
   * Ltensorflow/python/framework/ops/Tensor} that {@code tf.shape(y)} now allocates (post the
   * wala/ML#489 root-cause fix on this PR's `tensorflow.xml`); {@link
   * com.ibm.wala.cast.python.ml.client.BroadcastTo#getDefaultShapes}'s try/catch returns {@code
   * null} (lattice ⊤) instead of letting the exception abort the analysis. The result is shape ⊤
   * with dtype inherited from {@code x} (float32). Without the catch, analysis aborts and this test
   * fails &mdash; this is the direct regression guard for this PR's localized-tolerance fix.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/473">wala/ML#473</a>): the runtime answer is
   * (2, 3) of float32. When the helper learns to recognize {@code tf.shape(y)} as a shape arg
   * (rather than treating it as an unmodeled runtime tensor), tighten this assertion from {@link
   * #TENSOR_UNKNOWN_SHAPE_FLOAT32} to {@code TENSOR_2_3_FLOAT32}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBroadcastToRuntimeShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_broadcast_to_runtime_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.tensordot} with a scalar {@code axes}. Output
   * dtype is inherited from the {@code a} input (here float32); with {@code axes=1} the output
   * shape is {@code a.shape[:-1] + b.shape[1:]}, so two (2, 2) inputs yield (2, 2). See {@link
   * com.ibm.wala.cast.python.ml.client.Tensordot} (wala/ML#449).
   */
  @Test
  public void testTensordot()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensordot.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.trace}. Output dtype is inherited from the {@code
   * x} input (here float32); the output shape is the input shape with the last two dimensions
   * dropped, so a (2, 2) input yields a scalar. See {@link
   * com.ibm.wala.cast.python.ml.client.Trace} (wala/ML#449).
   */
  @Test
  public void testTrace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_trace.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.trace} on a batched input. The trace collapses the
   * last two dimensions, so a (3, 2, 2) input yields a (3,) output that inherits the input dtype.
   * Exercises the leading-dimensions path of {@link com.ibm.wala.cast.python.ml.client.Trace}
   * (wala/ML#449).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTraceBatched()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_trace_batched.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_3_2_2_FLOAT32),
            3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose(a)} with no {@code perm} reverses the axes, so a {@code (2,
   * 3)} input yields {@code (3, 2)}. Previously modeled as a first-argument {@code pass_through},
   * which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposeDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_default",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose(a, perm)} with a constant {@code perm} permutes the axes so
   * output axis {@code i} is input axis {@code perm[i]}: a {@code (2, 3, 4)} input with {@code perm
   * = [0, 2, 1]} yields {@code (2, 4, 3)}. Previously modeled as a first-argument {@code
   * pass_through}, which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposePerm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_perm",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_4_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose} resolves a {@code perm} passed as a tensor constant (rather
   * than a Python list literal): a {@code (2, 3, 4)} input with {@code perm = tf.constant([2, 1,
   * 0])} permutes precisely to {@code (4, 3, 2)}. Exercises the tensor-constant {@code perm}
   * resolution path of {@link com.ibm.wala.cast.python.ml.client.Transpose}, distinct from the
   * list-literal path. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket
   * 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposeTensorPerm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_tensor_perm",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.diag(diagonal)} increases rank by one: it places the input's
   * last axis on the diagonal of a new trailing square, so a {@code (4,)} input yields {@code (4,
   * 4)}. Previously modeled as a first-argument {@code pass_through}, which reported the input
   * shape unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDiag()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_diag",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.diag_part(input)} decreases rank by one: it extracts the
   * diagonal of each trailing square, so a {@code (3, 3)} input yields {@code (3,)}. Previously
   * modeled as a first-argument {@code pass_through}, which reported the input shape unchanged. See
   * <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDiagPart()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_diag_part",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.matrix_transpose(a)} swaps the last two axes, preserving leading
   * batch dimensions, so a {@code (2, 3)} input yields {@code (3, 2)}. Previously modeled as a
   * first-argument {@code pass_through}, which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMatrixTranspose()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_matrix_transpose",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.adjoint(matrix)} (conjugate transpose) swaps the last two axes
   * exactly like {@code matrix_transpose}, so a {@code (2, 3)} input yields {@code (3, 2)}.
   * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
   * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testAdjoint()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_adjoint",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.tile(input, multiples)} multiplies each axis by the corresponding entry
   * of {@code multiples}, so a {@code (2, 3)} input tiled by {@code [2, 1]} yields {@code (4, 3)}.
   * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
   * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTile()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tile.py", "consume_tile", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.solve(matrix, rhs)} reports the shape and dtype of the
   * right-hand side {@code rhs} (arg 1), not the coefficient {@code matrix} (arg 0). A {@code (3,
   * 3)} matrix and a {@code (3, 5)} rhs yield a {@code (3, 5)} result. Previously the op was
   * modeled as a first-argument {@code pass_through}, which reported the {@code (3, 3)} matrix
   * shape — an actively wrong answer. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_solve.py", "consume_solve", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.cholesky_solve(chol, rhs)} reports the shape and dtype of the
   * right-hand side {@code rhs} (arg 1), not the Cholesky factor {@code chol} (arg 0). See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCholeskySolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_solve.py", "consume_cholesky_solve", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.triangular_solve(matrix, rhs)} reports the shape and dtype of
   * the right-hand side {@code rhs} (arg 1), not the coefficient {@code matrix} (arg 0). See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTriangularSolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_solve.py",
        "consume_triangular_solve",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.sinh} (wala/ML#422). */
  @Test
  public void testSinh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sinh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.cosh} (wala/ML#422). */
  @Test
  public void testCosh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cosh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.acosh} (wala/ML#422). */
  @Test
  public void testAcosh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_acosh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.log1p} (wala/ML#422). */
  @Test
  public void testLog1p()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log1p.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.expm1} (wala/ML#422). */
  @Test
  public void testExpm1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expm1.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.round} (wala/ML#422). */
  @Test
  public void testRound()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_round.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.reciprocal} (wala/ML#422). */
  @Test
  public void testReciprocal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reciprocal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.square} (wala/ML#422). */
  @Test
  public void testSquare()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_square.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.erf} (wala/ML#422). */
  @Test
  public void testErf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_erf.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.erfc} (wala/ML#422). */
  @Test
  public void testErfc()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_erfc.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.maximum} (wala/ML#422). */
  @Test
  public void testMaximum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_maximum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.minimum} (wala/ML#422). */
  @Test
  public void testMinimum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_minimum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testMaximum}: {@code tf.math.maximum(x=..., y=...)}.
   * Exercises the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testMaximumKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_maximum_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testMinimum}: {@code tf.math.minimum(x=..., y=...)}.
   * Exercises the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testMinimumKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_minimum_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sqrt}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSqrt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sqrt.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Log}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLog()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Negative}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testNegative()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_negative.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sin}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Cos}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCos()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cos.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Floor}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testFloor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_floor.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sign}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sign.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.expand_dims(input, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.ExpandDims} generator overrides {@code getDefaultShapes} to
   * ⊤ pending an axis-aware shape composer. Replacing the stale {@code array_ops.expand_dims}
   * pass_through alias with the dedicated routing (this PR's earlier review fix) made the override
   * actually fire, so the assertion now sees the ⊤ output the override emits.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/481">wala/ML#481</a>): once the axis-aware
   * composer lands as a follow-up, tighten this from {@code TENSOR_UNKNOWN_SHAPE_FLOAT32} to {@code
   * (1, 3)} float32 (the precise insert-at-axis result).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testExpandDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expand_dims.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testExpandDims()} for {@code axis=-1}: trailing length-1 dim. Input shape
   * {@code (3,)} produces output shape {@code (3, 1)}.
   */
  @Test
  public void testExpandDimsAxisNeg1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expand_dims_axis_neg1.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_1_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.math.pow(x, y)}. Element-wise binary; output shape is the
   * broadcast of {@code x} and {@code y} (here both {@code (3,)}, so {@code (3,)}); output dtype
   * matches {@code x} (TF requires {@code x}/{@code y} to share dtype, so dtype-from-{@code x} is
   * sound). Routed through {@link com.ibm.wala.cast.python.ml.client.ElementWiseOperation}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testPow}: {@code tf.math.pow(x=x, y=y)}. Exercises the
   * keyword arg-resolution path through {@link
   * com.ibm.wala.cast.python.ml.client.ElementWiseOperation}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPowKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Mixed positional/keyword variant of {@link #testPow}: {@code tf.math.pow(x, y=y)}. Exercises
   * the case where the first argument is positional and the rest are keyword arguments.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPowMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testAbstractMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method.py", "D.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_abstract_method.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method2.py", "D.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_abstract_method2.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method3.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.meshgrid(*xi)}. Returns N tensors (one per input)
   * sharing the broadcast of input shapes and the first input's dtype. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Meshgrid} generator implements {@link
   * com.ibm.wala.cast.python.ml.client.TupleElementProvider}, but the XML only allocates one tuple
   * slot (field 0); the second meshgrid output ({@code Y} in the fixture) doesn't have a backing
   * alloc, so it falls through to ⊤/UNKNOWN and the aggregate union leaks the {@code
   * float32}/{@code unknown} pair. See <a href="https://github.com/wala/ML/issues/480">
   * wala/ML#480</a>.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands (or the
   * meshgrid XML is updated to allocate per-input tuple slots), narrow the assertion to {@code
   * Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMeshgrid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_meshgrid.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Companion to {@link #testMeshgrid}: 2-parameter sink {@code f(X, Y)} so the analyzer's
   * per-parameter typing on `tf.meshgrid`'s tuple result can be observed at distinct value numbers.
   * The two parameters split the leak asymmetrically:
   *
   * <ul>
   *   <li>vn=2 ({@code X}) shows the full {@code float32}/⊤-dtype union &mdash; the meshgrid XML
   *       only allocates field-0 of the tuple, so when {@code X} aliases that slot through PA
   *       propagation it picks up both the precise float32 alloc and the ⊤ tuple-slot leak.
   *   <li>vn=3 ({@code Y}) collapses to just the precise {@code float32} &mdash; the second
   *       meshgrid output's allocation site doesn't reach this parameter through the PA graph, so
   *       the ⊤ leak doesn't surface.
   * </ul>
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow
   * vn=2's assertion to {@code Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)} (vn=3 is already there).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMeshgridXY()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_meshgrid_xy.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE),
            3, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Generator-dispatch test for the 3-arg form of {@code tf.where(condition, x, y)}. The dedicated
   * {@link com.ibm.wala.cast.python.ml.client.Where} generator produces shape and dtype by unioning
   * the inferred sets over {@code x} and {@code y} (and intentionally ignoring {@code condition}'s
   * shape, per the modeling note in {@code Where}'s class Javadoc). The fixture has all three
   * operands shape {@code (3,)} float32, so the union collapses to the singleton {@code (3,)
   * float32}. Fix for the wala/ML#422 listing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWhere()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_where.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Sibling of {@link #testWhere()} that exercises the union-over-{@code x}-and-{@code y} path with
   * broadcast-compatible different shapes: {@code x} is {@code (3,)} float32 and {@code y} is
   * {@code (2, 3)} float32. The runtime broadcast result is {@code (2, 3)}; the static analysis
   * unions the two operand shapes to {@code {(3,), (2, 3)}}, which is sound but imprecise.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/482">wala/ML#482</a>): once broadcast-shape
   * composition lands, tighten this assertion from the union {@code {TENSOR_3_FLOAT32,
   * TENSOR_2_3_FLOAT32}} to the precise {@code TENSOR_2_3_FLOAT32}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWhereBroadcast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_where_broadcast.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32, TENSOR_2_3_FLOAT32)));
  }

  /**
   * Tier-6 op (wala/ML#449): {@code tf.sort(values, ...)}. The XML routes the call through {@code
   * convert_to_tensor} of {@code values}, so shape and dtype pass through unchanged — no dedicated
   * generator needed.
   */
  @Test
  public void testSort()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sort.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.tensor_scatter_nd_update}. Output dtype AND shape are
   * inherited from the {@code tensor} input — true shape-and-dtype passthrough on the first arg.
   * Here the input is shape {@code (4,)} float32, so the precise expected result is {@code (4,)
   * float32}. See {@link com.ibm.wala.cast.python.ml.client.TensorScatterNdUpdate} (wala/ML#449).
   *
   * <p>Post-master-merge of wala/ML#380's `tensor_scatter_nd_update` inlining (#237), the analysis
   * also produces an additional fully-⊤ context — likely a context where the input arg's PTS isn't
   * recovered through the inlined synthetic body, so the passthrough returns ⊤/UNKNOWN. The
   * assertion captures both contexts (per the prefer-observed-assertion convention from
   * `CONTRIBUTING.md`); a precision improvement that eliminates the ⊤ context will narrow the
   * actual to just {@code TENSOR_4_FLOAT32} and this test will fail with a clear "expected union,
   * got per-context" diff — that's the cue to update.
   *
   * <p>TODO(wala/ML#474): Once the additional fully-⊤ context for the inlined-passthrough path is
   * investigated/eliminated, narrow the assertion to {@code Set.of(TENSOR_4_FLOAT32)}.
   */
  @Test
  public void testTensorScatterNdUpdate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tensor_scatter_nd_update.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Generator-dispatch test for {@code tf.sequence_mask}. With no {@code dtype} argument the output
   * dtype is the TF-default {@code bool}; with a constant {@code maxlen} the shape is {@code
   * lengths.shape + [maxlen]}, so {@code sequence_mask([1, 3, 2], maxlen=5)} yields (3, 5).
   * (wala/ML#449 Tier 8.)
   */
  @Test
  public void testSequenceMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sequence_mask.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_5_BOOL)));
  }

  /**
   * Generator-dispatch test for {@code tf.sequence_mask} with an explicit {@code dtype} override
   * ({@code tf.sequence_mask(..., maxlen=5, dtype=tf.int32)}). The output dtype follows the
   * argument ({@code int32}) rather than the default {@code bool}, and the constant {@code maxlen}
   * gives the precise (3, 5) shape. Regression guard for surfacing the {@code dtype} parameter
   * through {@link com.ibm.wala.cast.python.ml.client.SequenceMask}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSequenceMaskDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sequence_mask_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_5_INT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.nn.embedding_lookup}. Output dtype is inherited from the
   * {@code params} input (here float32); the output shape is {@code ids.shape + params.shape[1:]},
   * so a (3, 2) table looked up by (2,) ids yields (2, 2). See {@link
   * com.ibm.wala.cast.python.ml.client.EmbeddingLookup} (wala/ML#449 Tier 8).
   */
  @Test
  public void testEmbeddingLookup()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_embedding_lookup.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.boolean_mask}. Output dtype is inherited from the {@code
   * tensor} input (here float32); the masked axis collapses to a dynamic dimension (the runtime
   * {@code True} count), so masking a (3, 2) tensor with a rank-1 mask yields {@code [Dynamic, 2]}.
   * The leading dimension is inherently runtime, so it stays dynamic. See {@link
   * com.ibm.wala.cast.python.ml.client.BooleanMask} (wala/ML#449 Tier 8).
   */
  @Test
  public void testBooleanMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_boolean_mask.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.image.extract_patches}. Output dtype is inherited from
   * the {@code images} input (here float32); the output shape is {@code [batch, out_rows, out_cols,
   * sizes_r * sizes_c * channels]}, so a (1, 10, 10, 3) image with {@code sizes=[1, 3, 3, 1]},
   * {@code strides=[1, 5, 5, 1]}, {@code rates=[1, 1, 1, 1]} and {@code VALID} padding yields (1,
   * 2, 2, 27). See {@link com.ibm.wala.cast.python.ml.client.ExtractPatches} (wala/ML#449 Tier 8).
   */
  @Test
  public void testExtractPatches()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_extract_patches.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_2_27_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.image.extract_patches} called with a Python <em>list
   * literal</em> {@code images} argument (rather than a {@code tf.Tensor}), per <a
   * href="https://github.com/wala/ML/issues/584">wala/ML#584</a>. The list-literal shape {@code (1,
   * 1, 1, 1)} is recovered from the nesting structure, and the result is the concrete {@code (1, 0,
   * 0, 9) int32}: a {@code 3x3} patch does not fit the {@code 1x1} image, so {@code VALID} padding
   * yields a 0-extent spatial output (depth {@code 3*3*1 = 9}), matching the runtime shape the
   * Python fixture asserts (<a href="https://github.com/wala/ML/issues/585">wala/ML#585</a>).
   */
  @Test
  public void testExtractPatches2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_extract_patches2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_0_0_9_INT32)));
  }

  /**
   * Regression guard for {@code tf.image.extract_patches} called with an {@code images} argument
   * built from a nested list <em>comprehension</em> (the wala/ML#584 corpus case), per <a
   * href="https://github.com/wala/ML/issues/584">wala/ML#584</a>. Resolving such an {@code images}
   * operand throws inside the generator; before the fix that aborted the whole type computation and
   * the result dropped its tensor classification entirely. The result must still be recognized as a
   * tensor — here ⊤ shape and ⊤ dtype, since a comprehension's computed elements (unlike a
   * literal's constants) yield no statically inferable dtype.
   */
  @Test
  public void testExtractPatchesComprehension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_extract_patches_comprehension.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /** Pure-passthrough generator test for {@code tf.math.tan} (wala/ML#422). */
  @Test
  public void testTan()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tan.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.asin} (wala/ML#422). */
  @Test
  public void testAsin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_asin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.atan} (wala/ML#422). */
  @Test
  public void testAtan()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.asinh} (wala/ML#422). */
  @Test
  public void testAsinh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_asinh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.atanh} (wala/ML#422). */
  @Test
  public void testAtanh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atanh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.softplus} (wala/ML#422). */
  @Test
  public void testSoftplus()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_softplus.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.softsign} (wala/ML#422). */
  @Test
  public void testSoftsign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_softsign.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.atan2} (wala/ML#422). */
  @Test
  public void testAtan2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testAtan2}: {@code tf.math.atan2(y=..., x=...)}. Exercises
   * the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testAtan2Kw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan2_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator test for {@code tf.strings.as_string} on a 2-arg sink {@code f(y, x)}. Output shape
   * is the input's shape (here {@code (3,)}); output dtype is fixed to {@code string}
   * (wala/ML#422). Asserts that both the {@code as_string} output (`y`, string dtype) at vn=2 and
   * its input (`x`, float32) at vn=3 classify precisely &mdash; the multi-tensor-sink pattern
   * doesn't break classification on either flow when run inside the full test suite.
   */
  @Test
  public void testAsString2ArgSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_as_string_2arg_sink.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_3_STRING), 3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Sibling of {@link #testAsString2ArgSink()} with two 1-arg sinks {@code f(y)} and {@code g(x)},
   * asserting the {@code y}-side sink. The {@code as_string} output's string-dtype tensor
   * classifies precisely at vn=2.
   */
  @Test
  public void testAsStringTwoSinksY()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_as_string_two_sinks.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_STRING)));
  }

  /**
   * Companion to {@link #testAsStringTwoSinksY()} asserting the {@code x}-side sink {@code g(x)}.
   * The {@code x} input classifies precisely as float32 at vn=2.
   */
  @Test
  public void testAsStringTwoSinksX()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_as_string_two_sinks.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  @Test
  public void testGradient2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/464">wala/ML#464</a>: when
   * {@code sources} is a list (the common Keras pattern), {@code tape.gradient} returns a parallel
   * list of fresh tensors and {@code grads[i]} must resolve to the shape/dtype of the i-th source.
   * The fixture passes both {@code grads[0]} (for {@code w1}, a {@code [2]}-shaped float32) and
   * {@code grads[1]} (for {@code w2}, a {@code [1, 1]}-shaped float32) to {@code f} across two
   * separate calls; with the {@link com.ibm.wala.cast.python.ml.client.Gradient} {@code
   * TupleElementProvider} implementation, {@code f}'s parameter resolves to the union of {@link
   * #TENSOR_2_FLOAT32} and {@link #TENSOR_1_1_FLOAT32}.
   */
  @Test
  public void testGradientList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gradient_list.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_1_1_FLOAT32)));
  }

  /**
   * Tighter variant of {@link #testGradientList()}: passes both gradients in a single {@code
   * f(grads[0], grads[1])} call, so the analyzer must resolve each argument's tensor type
   * independently per its source index rather than as a union across two call sites. {@code f}'s
   * first parameter (vn=2) must resolve to {@link #TENSOR_2_FLOAT32} (from {@code w1}) and the
   * second parameter (vn=3) to {@link #TENSOR_1_1_FLOAT32} (from {@code w2}). Closes part of <a
   * href="https://github.com/wala/ML/issues/464">wala/ML#464</a>.
   */
  @Test
  public void testGradientList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gradient_list2.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_1_1_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.cast(x, dtype)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Cast} generator extends {@link
   * com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator} for shape and overrides the
   * dtype-arg position to point at {@code dtype}; the {@code tf.cast} {@code pass_through} alias
   * that previously bypassed the override was removed in <a
   * href="https://github.com/wala/ML/issues/499">wala/ML#499</a>, so the static analysis now
   * reports the explicit cast target ({@code int32}) rather than the input's dtype ({@code
   * float32}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cast.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.clip_by_value(t, clip_value_min, clip_value_max)}. Pure
   * passthrough — output shape and dtype both inherit from {@code t}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testClipByValue()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_clip_by_value.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.nn.leaky_relu(features)}. Pure passthrough — output shape
   * and dtype both inherit from {@code features} (the input tensor). Mirrors {@link #testRelu()};
   * the {@link com.ibm.wala.cast.python.ml.client.LeakyRelu} generator extends {@link
   * com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLeakyRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_leaky_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Tier-1 generator (wala/ML#449): {@code tf.identity(input)} returns a fresh tensor with the same
   * shape and dtype as {@code input}. Pre-fix this routed via {@code identity}'s synthetic XML
   * which ultimately allocates a {@code Ltensorflow/python/framework/ops/convert_to_tensor} —
   * {@link com.ibm.wala.cast.python.ml.client.ConvertToTensor} handles dtype/shape, but the extra
   * indirection through {@code identity}'s wrapper class can be tightened.
   */
  @Test
  public void testIdentity()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_identity.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Companion to {@link #testIdentity()} exercising the keyword-argument call site {@code
   * tf.identity(input=...)}. The arg-resolution helpers in {@link
   * com.ibm.wala.cast.python.ml.client.Identity} (and the underlying {@code
   * getArgumentPointsToSet(builder, paramPos, paramName)}) resolve keyword args via {@code
   * paramName}; without a kwarg fixture that branch is dead-on-arrival in the test data.
   */
  @Test
  public void testIdentityKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_identity_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tier-1 generator (wala/ML#449): {@code tf.stop_gradient(input)} returns a fresh tensor with the
   * same shape and dtype as {@code input}. Pre-fix this routed through {@code ReadDataFallback}
   * (the alloc has no value/dtype field bindings, just a {@code read_data} marker) and emitted
   * {@code [{? of unknown}]}; the {@link com.ibm.wala.cast.python.ml.client.StopGradient} generator
   * now reads {@code input}'s shape/dtype directly via the same {@code shapesOfArg} / {@code
   * dtypesOfArg} pattern as {@link com.ibm.wala.cast.python.ml.client.Sigmoid}.
   */
  @Test
  public void testStopGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stop_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Companion to {@link #testStopGradient()} exercising the keyword-argument call site. */
  @Test
  public void testStopGradientKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stop_gradient_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Lock-in test for wala/ML#449 Tier-1 coverage of {@code tf.nn.bias_add(value, bias)}: returns a
   * fresh tensor with the same shape and dtype as {@code value} (bias is broadcast-added but
   * doesn't change the receiver's shape). Unlike {@link #testIdentity()} / {@link
   * #testStopGradient()} — which got dedicated {@link com.ibm.wala.cast.python.ml.client.Identity}
   * / {@link com.ibm.wala.cast.python.ml.client.StopGradient} generators paired with a direct
   * {@code <new>+<return>} in the XML — {@code bias_add} is modeled in {@code tensorflow.xml} as a
   * delegation: {@code <new>} of a {@code convert_to_tensor} function-object, {@code <call>} of its
   * {@code do} with {@code value} as the argument, then {@code <return>} of that result (no
   * dedicated Java generator; the actual tensor allocation happens inside {@code
   * convert_to_tensor.do()}). That delegation suffices for the shape/dtype-passthrough semantics
   * this test exercises.
   */
  @Test
  public void testBiasAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bias_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testBiasAdd()} exercising the all-keyword call site {@code
   * tf.nn.bias_add(value=..., bias=...)}.
   */
  @Test
  public void testBiasAddKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bias_add_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Diagnostic for the existing {@code add_n} XML pattern (read list field 0 → {@code
   * convert_to_tensor}). Captures the observable static-analysis output for {@code tf.add_n([t1,
   * t2])} where both list elements are {@code tf.constant} tensors. If this test passes with {@code
   * TENSOR_3_INT32}, the list-element-PTS path through {@code <getfield class="Llist" field="0">}
   * works and Tier 5 ops ({@code concat}/{@code stack}/{@code meshgrid}) can use the same pattern
   * cheaply. If it produces {@code ? of unknown}, the pattern doesn't propagate types and Tier 5
   * needs Java-side list-element traversal logic.
   */
  @Test
  public void testAddN()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_n.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.concat(values, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Concat} generator computes the precise output shape by
   * walking every entry in the {@code values} list, summing each input's dim along the resolved
   * {@code axis}, and inheriting the rest of the shape from the first input. The fixture
   * concatenates two {@code (3,)} tensors along {@code axis=0}, so the precise output is {@code
   * (6,)}; dtype is inherited from the first element ({@code int32}).
   */
  @Test
  public void testConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_INT32)));
  }

  /**
   * Multi-rank {@code tf.concat([t1, t2], axis=1)} with {@code (2, 3)} inputs. Exercises the
   * rank-aware path in {@link com.ibm.wala.cast.python.ml.client.Concat#computeConcatenatedShape}:
   * non-axis dim preservation (the leading {@code 2} survives) and the axis-dim sum (the trailing
   * {@code 3 + 3 = 6}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testConcatMultirank()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat_multirank.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_6_INT32)));
  }

  /**
   * {@code tf.concat([t1, t2], axis=-1)} with {@code (2, 3)} inputs. Exercises the negative-axis
   * normalization in {@link com.ibm.wala.cast.python.ml.client.Concat#computeConcatenatedShape}:
   * {@code axis = -1} resolves to {@code rank - 1 = 1} for rank-2 inputs, producing the same {@code
   * (2, 6)} answer as the explicit {@code axis=1} fixture.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testConcatNegativeAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat_negaxis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_6_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.stack(values, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Stack} generator computes the precise output shape by
   * reading the {@code values} list's PTS-derived length {@code N} and inserting it at the resolved
   * {@code axis} position into the first element's shape: {@code values[0].shape[:axis] + (N,) +
   * values[0].shape[axis:]}. The fixture stacks two {@code (3,)} tensors with {@code axis=0}, so
   * the precise output is {@code (2, 3)}; dtype is inherited from the first element ({@code
   * int32}).
   */
  @Test
  public void testStack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stack.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * Distilled regression guard for {@code tf.matmul}'s batched form (wala/ML#718): the leading
   * (batch) dimensions carry through and the trailing two dimensions compose as the matrix product,
   * so the rank is preserved. The analysis previously collapsed every product to rank two.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBatchedMatMul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_batched_matmul.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 5))));
  }

  /**
   * NLPGNN's {@code WDEmbedding} in miniature (wala/ML#711, wala/ML#717): the output reshape's
   * target slices the rank-conditional {@code tf.expand_dims} guard's φ, so the composition
   * cross-products the two source shapes per position: four members, including the runtime {@code
   * (2, 2, 8)}, with the trailing dimension static in every member. The mixed-pairing members are
   * the cross-product's sound over-approximation; pairing the evaluation per φ member would drop
   * them. The embedding table's {@code float32} survives through both the {@code gather} arm and
   * the {@code one_hot}/{@code matmul} arm.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmbeddingOutput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_output.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 16),
                TensorType.of(FLOAT_32, 2, 8),
                TensorType.of(FLOAT_32, 2, 2, 16),
                TensorType.of(FLOAT_32, 2, 2, 8))));
  }

  /**
   * Opaque-size variant of {@link #testEmbeddingOutput()} (wala/ML#717), mirroring the vendored
   * NLPGNN embedding, whose table size comes from a checkpoint config the analysis cannot read. The
   * output reshape's trailing element {@code input_shape[-1] * self.embedding_size} then has an
   * unresolvable factor, but arithmetic over a shape-vector subscript is one scalar dimension, so
   * the value degrades to dynamic while the rank and the leading dimensions survive, per guard-φ
   * member.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmbeddingDynamicSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_dynamic_size.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(2), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(2), new NumericDim(2), UnresolvedDim.INSTANCE)))));
  }

  /**
   * Exercise the duck-typed {@code numpy.ndarray.astype(...)} dispatch path added for wala/ML#356.
   * The receiver is {@code x_train}, the first element of {@code
   * tf.keras.datasets.mnist.load_data()}. Without mnist modeling, the receiver's concrete shape
   * cannot be resolved; after the {@code astype} call, {@code consume}'s parameter is recognised as
   * a float32 tensor with a {@code null} dims list.
   */
  @Test
  public void testAstype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_astype.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_60000_28_28_FLOAT32)));
  }

  /**
   * Regression guard for wala/ML#403: chained {@code x.astype(int32).astype(float32)} on an mnist
   * receiver. The first cast's result is a synthetic-method return whose PointerKey is implicit, so
   * the receiver-shape lookup for the second {@code astype} call hits the {@code
   * IllegalArgumentException}-catch fallback path in {@link AstypeOperation#getDefaultShapes},
   * which returns {@code null} (⊤) for shape while dtype still resolves to {@code float32}.
   *
   * <p>The runtime-vs-analyzer asymmetry here ({@code (60000, 28, 28) float32} at runtime vs.
   * {@code TensorType(float32, null)} from the analyzer) is the same kind of deliberate limitation
   * as {@link #testInputUnresolvableShape}: traversing implicit-PK chains across synthetic-method
   * returns is a known architectural gap (wala/ML#402 / wala/WALA#1889), and returning ⊤ rather
   * than ⊥ is the lattice-correct response in the meantime — dtype still carries through, so
   * downstream analysis isn't dropped. The test asserts the analyzer's lattice-correct output
   * rather than suppressing it as a would-be-fixed failure; if and when the implicit-PK chain
   * traversal lands, this expectation flips to {@code TENSOR_60000_28_28_FLOAT32} as part of that
   * change.
   */
  @Test
  public void testAstypeChained()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_astype_chained.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Exercise {@link NdarrayReshape#getDefaultShapes}'s SSA-substrate DU walk: resolves the {@code
   * -1} in {@code x_train.reshape([-1, 784])} by tracing the receiver back to {@code mnist.x_train}
   * ({@code (60000, 28, 28)}). Guards {@link TensorGenerator#getShapesOrSSAChain} against
   * regression.
   *
   * <p>Dtype propagation ({@code uint8}) uses the existing {@code getDTypes(builder, receiverVn)}
   * path, which resolves through the normal PA because {@code x_train}'s dtype is carried on the
   * {@link MnistInputData}-manufactured allocation.
   */
  @Test
  public void testNdarrayReshape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ndarray_reshape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_60000_784_UINT8)));
  }

  /**
   * A constant transpose permutation fixes its unresolved input's rank (<a
   * href="https://github.com/wala/ML/issues/734">wala/ML#734</a>): {@code inputs} arrives
   * shape-opaque with a proven dtype (a cast of an unmodeled {@code tf.roll} result), and the
   * explicit-perm {@code tf.transpose(inputs, [1, 0, 2])} proves it rank-3, the {@code crf_forward}
   * pattern. Runtime-verified {@code (4, 3, 5) float32}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransposeRankRecovery()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose_rank.py",
        "f",
        1,
        2,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)))));
  }
}
