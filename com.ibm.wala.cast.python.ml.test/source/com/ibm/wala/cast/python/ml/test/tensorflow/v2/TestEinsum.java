package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of {@code tf.einsum} shape and dtype inference, carved from the {@link
 * TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestEinsum extends AbstractTensorTest {

  /**
   * Generator test for {@code tf.einsum(equation, *inputs)}. The {@link
   * com.ibm.wala.cast.python.ml.client.Einsum} generator parses the equation string and composes
   * the output shape from each input's shape. Here {@code einsum("ij,jk->ik", a, b)} with {@code a}
   * and {@code b} both {@code (2, 2)} yields {@code (2, 2)}. Output dtype inherits from the first
   * tensor input ({@code float32}). See wala/ML#507.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Implicit-output einsum: with no {@code ->}, the output labels are those occurring exactly once,
   * in alphabetical order, so {@code "ij,jk"} composes the same {@code (2, 2)} shape as {@code
   * "ij,jk->ik"}. See wala/ML#507.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumImplicit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_implicit.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Broadcasting-ellipsis einsum (wala/ML#705): each input's {@code ...} binds the axes its letters
   * don't consume (here none), the groups broadcast, and the output's {@code ...} receives the
   * result, composing the same {@code (2, 2)} shape as {@code "ij,jk->ik"}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumEllipsis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_ellipsis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Broadcasting-ellipsis einsum with real batch axes (wala/ML#705): the ellipsis groups {@code (2,
   * 1)} and {@code (5,)} broadcast right-aligned to {@code (2, 5)}, which the output's {@code ...}
   * receives ahead of the {@code ik} letters.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumBatchBroadcast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_batch.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5, 3, 2))));
  }

  /**
   * Diagonal einsum (a repeated label within one term, {@code "ii->i"}, wala/ML#705): the repeated
   * label names axes the runtime requires equal, so its occurrences refine one another and the
   * output label composes the single {@code (2,)} diagonal dimension.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumDiagonal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_diag.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * Trace einsum ({@code "ii"} in implicit mode, wala/ML#705): the repeated label drops from the
   * implicit output, contracting the diagonal to a scalar.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumTrace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_trace.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Batch-diagonal einsum ({@code "...ii->...i"}, wala/ML#705): the ellipsis and the repeated label
   * compose, keeping the batch axis and extracting the diagonal of each square slice.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumBatchDiagonal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_trace.py", "g", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2))));
  }

  /**
   * Malformed and unsatisfiable einsum equations (wala/ML#705): a stray or truncated dot, a second
   * ellipsis in one term, a non-letter label, a malformed output term, a repeated output label, a
   * label/rank mismatch, non-broadcastable batch axes, broadcast axes with no output ellipsis, and
   * the empty equation all compose no shape, keeping the sound ⊤ fallback with a precise dtype.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumMalformed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_malformed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Ellipsis identity with leading letters ({@code "i...j->i...j"}, wala/ML#705): labels before the
   * ellipsis bind leading axes and labels after it bind trailing ones.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumEllipsisLeadingLetter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_mixed.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 4))));
  }

  /**
   * Implicit mode with an ellipsis ({@code "i...j"}, wala/ML#705): the broadcast group precedes the
   * once-occurring labels, composing {@code (3, 2, 4)} from a {@code (2, 3, 4)} input.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumImplicitEllipsis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_mixed.py", "g", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2, 4))));
  }

  /**
   * The mirror of {@link #testEinsumBatchBroadcast()} (wala/ML#705): the size-1 axis sits in the
   * second input's ellipsis group, so the {@code (5,)} and {@code (2, 1)} groups still broadcast to
   * {@code (2, 5)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumBatchBroadcastReversed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_mixed.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5, 3, 2))));
  }

  /**
   * A statically-unknown batch axis ({@code keras.Input}'s {@code None}) refines against a known
   * one (wala/ML#705): the runtime requires the unknown to be 1 or equal, so the known non-1 size
   * wins in either input order.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumSymbolicBatchRefined()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_mixed.py", "k", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * Two statically-unknown batch axes (wala/ML#705): the broadcast keeps the rank but not the size,
   * composing a rank-1 shape whose axis stays dynamic.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumSymbolicBatchPair()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_mixed.py",
        "m",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Three-term einsum operand refinement (wala/ML#704): the first operand binds {@code h} dynamic
   * ({@code keras.Input}'s {@code None}) and {@code n} to 3, the second operand's known {@code h =
   * 6} refines the dynamic binding, and the unresolved third operand's {@code "nq"} term proves
   * rank 2 with the shared {@code n = 3} and an {@link UnresolvedDim} {@code q}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumOperandRefinement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_operand_refinement.py",
        "project",
        3,
        4,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(new NumericDim(3), UnresolvedDim.INSTANCE))),
            3,
            Set.of(TENSOR_NONE_3_FLOAT32),
            4,
            Set.of(TensorType.of(FLOAT_32, 2, 6))));
  }

  /**
   * A multi-shape einsum operand is constrained too (wala/ML#704): the guard-φ argument carries
   * both {@code keras.Input} members, and each fills its non-numeric leading axis from the
   * constraint's known {@code n = 3} while keeping its own trailing size.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumOperandRefinement2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_operand_refinement.py",
        "fill",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 3, 5), TensorType.of(FLOAT_32, 3, 7)),
            3,
            Set.of(TensorType.of(FLOAT_32, 3, 2))));
  }

  /**
   * An unresolved einsum operand whose term carries an ellipsis proves no rank (wala/ML#704), so
   * the equation constrains nothing about it and the parameter keeps its unknown shape: the sound
   * bail.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumOperandRefinement3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_operand_refinement.py",
        "blur",
        2,
        3,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32), 3, Set.of(TensorType.of(FLOAT_32, 3, 5))));
  }

  /**
   * Two einsum call sites once proved disagreeing constraints for the same operand (wala/ML#704),
   * but the guard's flag is the constant {@code True} at the call, so the rank-3 site is dead and
   * no longer contributes (wala/ML#763): the surviving site's constraint asserts the runtime-true
   * rank 2 with the trailing 3. {@link #testEinsumOperandRefinement5()} keeps the disagreement
   * scenario alive on an opaque flag.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumOperandRefinement4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_operand_refinement.py",
        "choose",
        3,
        5,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(3)))),
            3,
            Set.of(TensorType.of(FLOAT_32, 3, 4)),
            4,
            Set.of(TensorType.of(FLOAT_32, 5, 6))));
  }

  /**
   * Two einsum call sites prove disagreeing constraints for the same operand (wala/ML#704), rank 2
   * with a trailing 3 versus rank 3 with a trailing 5, and the guard's flag is an opaque
   * environment read no fold decides (wala/ML#763), so neither is asserted and the parameter keeps
   * its unknown shape: the conflicting refinement drops.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumOperandRefinement5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_operand_refinement.py",
        "choose2",
        3,
        5,
        Map.of(
            2,
            Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32),
            3,
            Set.of(TensorType.of(FLOAT_32, 3, 4)),
            4,
            Set.of(TensorType.of(FLOAT_32, 5, 6))));
  }

  /**
   * NLPGNN's {@code einsum_via_matmul} matmul path in miniature, end to end (wala/ML#704): the
   * {@code get_shape_list} hop, the negated-parameter slice bounds, the {@code np.prod} folds, and
   * the {@code batch_dims + outer_dims} concatenation (wala/ML#708) all resolve, so the reshape arm
   * types as the precise runtime {@code (2, 4, 3, 5)}. The {@code (2, 4, 15)} member is the other
   * arm of the {@code len(outer_dims) > 1} guard's φ: the raw matmul result, which flows statically
   * even though the runtime always takes the reshape arm here, and which the batched {@code
   * tf.matmul} semantics type as the runtime-true intermediate {@code matmul((2, 4, 6), (6, 15))}
   * (wala/ML#718; previously the rank-collapsed artifact {@code (4, 5)}). The {@code (2, 4, 5)}
   * member is the same φ's other pairing: {@code w}'s points-to set carries both its defs (the
   * original {@code (6, 3, 5)} and the {@code len(w_shape) > 2} arm's reshape to {@code (6, 15)}),
   * so the batched product also composes against the un-reshaped def, taking its trailing {@code
   * 5}. Both extras are path-insensitive unions, not miscomputations.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumViaMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_via_matmul.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4, 3, 5),
                TensorType.of(FLOAT_32, 2, 4, 15),
                TensorType.of(FLOAT_32, 2, 4, 5))));
  }

  /**
   * The two-inner-dims variant of {@link #testEinsumViaMatmul()} (NLPGNN's {@code DenseLayer3dProj}
   * shape, {@code einsum_via_matmul(input_tensor, w, 2)}): exercises the {@code batch_dims +
   * [inner_dim]} concatenation of a shape vector with a literal list whose element is an {@code
   * np.prod} fold (wala/ML#708). The result types exactly the precise runtime {@code (2, 4, 6)}:
   * the batched {@code tf.matmul} semantics (wala/ML#718) eliminated the rank-collapsed {@code (3,
   * 6)} artifact this test once pinned, and the constant-decidable arm pruning (wala/ML#746)
   * eliminated the {@code (2, 4, 3, 6)} phantom that composed against the {@code num_inner_dims >
   * 1} guard's runtime-untaken un-reshaped arm.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumViaMatmul2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_via_matmul2.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 6))));
  }

  /**
   * Partial composition when an operand does not resolve (<a
   * href="https://github.com/wala/ML/issues/737">wala/ML#737</a>): {@code
   * tf.einsum("BFND,NDH->BFH", x, w)} with {@code x}'s shape opaque (a computed, non-literal
   * reshape list) and {@code w = (3, 5, 6)} statically known still proves the output rank and the
   * {@code H = 6} axis; the axes only {@code x} names carry {@link UnresolvedDim}. Runtime-verified
   * {@code (2, 4, 6) float32}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEinsumPartialOperand()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_partial.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(6))))));
  }
}
