package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_4_FLOAT32;
import static java.util.Arrays.asList;

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
 * Tests of dense-layer shape and dtype inference ({@code Dense*}/{@code Dense3d*}), carved from the
 * {@code TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestDenseLayers extends AbstractTensorTest {

  /**
   * NLPGNN's {@code DenseLayer3d} einsum path in miniature (wala/ML#704): the weight is built flat
   * from configuration fields, reshaped to rank 3 in {@code call} (the {@code hidden} leading dim
   * stays dynamic since {@code build}'s {@code input_shape} subscript doesn't resolve), and
   * consumed by {@code einsum("BFH,HND->BFND", ...)}. The parser refines the contracted {@code H}
   * label's dynamic occurrence with the input's known {@code 6} and composes the precise runtime
   * {@code (2, 4, 3, 5)}: the trailing head dims are static, per the issue's expectation.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dEinsum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_einsum.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 5))));
  }

  /**
   * NLPGNN's {@code DenseLayer3dProj} (wala/ML#704): the input arrives with an unresolvable shape
   * (the wala/ML#711 opaque-bound reshape), but {@code einsum("BFND,NDH->BFH", input_tensor, w)}
   * proves the input is rank 4, and the shared {@code N}/{@code D} labels take their extents from
   * the reshaped weight's statically-known {@code (3, 5, 6)}. The {@code B}/{@code F} axes carry no
   * {@code None} evidence, so they are {@code Unresolved} (wala/ML#721), not {@code Dynamic}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dProj()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_proj.py",
        "DenseLayer3dProj.call",
        1,
        4,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        new NumericDim(3),
                        new NumericDim(5))))));
  }

  /**
   * Operand-order companion of {@link #testDense3dEinsum()} (wala/ML#704): the weight, whose
   * contracted dim is dynamic, comes first in the equation ({@code "HND,BFH->BFND"}), so the
   * input's statically-known occurrence of the shared {@code H} label arrives second and refines
   * the earlier dynamic one. The composed shape is the same precise {@code (2, 4, 3, 5)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dEinsum2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_einsum2.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 5))));
  }

  /**
   * Composition of {@link #testDense3dEinsum()} and {@link #testEinsumViaMatmul()} (wala/ML#704):
   * NLPGNN's {@code DenseLayer3d.call} delegates to the module-level {@code einsum_via_matmul}
   * helper. The {@code input_tensor} parameter carries the precise type across the layer-call
   * boundary, and the {@code w} parameter is fully static: the trailing dimensions are the folded
   * configuration fields, the leading (hidden) dimension resolves through {@code build}'s {@code
   * input_shape} subscript ({@code self.hidden_size = input_shape[2]}, wala/ML#712), and the
   * literal-shape pin agrees with the generator-side result (wala/ML#713), so the union is a single
   * member.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(TensorType.of(FLOAT_32, 6, 3, 5))));
  }

  /**
   * Escalation of {@link #testDense3dMatmul()} to NLPGNN's real dispatch shape (wala/ML#704): the
   * {@code DenseLayer3d} is created in an outer attention layer's {@code build}, stored as an
   * attribute, and invoked through it in the outer {@code call} — two trampoline hops before {@code
   * einsum_via_matmul} sees the input tensor. The {@code input_tensor} parameter carries the
   * precise type through both hops, and the {@code w} parameter resolves exactly as in the one-hop
   * composition, including the hidden dimension through {@code build}'s {@code input_shape}
   * subscript (wala/ML#712).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul2.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(TensorType.of(FLOAT_32, 6, 3, 5))));
  }

  /**
   * Negative-index companion of {@link #testDense3dMatmul()} (wala/ML#712): {@code build} stores
   * {@code input_shape[-1]}, exercising the Python negative-index arm of the shape-vector subscript
   * resolution.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul3.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(TensorType.of(FLOAT_32, 6, 3, 5))));
  }

  /**
   * Explicit-{@code build} companion of {@link #testDense3dMatmul()} (wala/ML#712): the script
   * calls {@code layer.build(x.shape)} before the layer call, so the {@code input_shape} parameter
   * also receives a real argument, exercising the explicit-caller arm of the walk alongside the
   * lazy-build trampoline hop.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul4.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(TensorType.of(FLOAT_32, 6, 3, 5))));
  }

  /**
   * Two-instance guard for the {@code build} subscript resolution (wala/ML#712): the class is
   * instantiated on inputs with different hidden sizes, so the per-class attribute chase sees
   * conflicting {@code hidden_size} stores and must bail; the weight's leading dimension stays
   * dynamic (a sound conservative result) while the folded configuration fields keep their
   * constants.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul5.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6), TensorType.of(FLOAT_32, 2, 4, 8)),
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, new NumericDim(3), new NumericDim(5))))));
  }

  /**
   * Arithmetic-over-subscript companion of {@link #testDense3dMatmul()} (wala/ML#714): {@code
   * build} derives the head size the way NLPGNN's {@code ALBERTAttention.build} does, dividing a
   * shape subscript by a configuration field under an {@code int(...)} wrapper ({@code
   * self.head_size = int(input_shape[2]/self.num_attention_heads)}), so the reshaped weight's
   * dimensions all fold.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul6.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(TensorType.of(FLOAT_32, 6, 3, 2))));
  }

  /**
   * Bare-division guard companion of {@link #testDense3dMatmul6()} (wala/ML#714): a branch the
   * runtime never takes stores {@code input_shape[2] / 4}, a bare float division the fold must
   * refuse (its non-exact truncation is undefined without the {@code int(...)} wrapper), so the
   * attribute is unresolvable and the weight's trailing dimension soundly stays dynamic.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul7.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(6), new NumericDim(3), UnresolvedDim.INSTANCE)))));
  }

  /**
   * Nested-arithmetic companion of {@link #testDense3dMatmul6()} (wala/ML#714): the divisor is
   * itself an arithmetic expression over a configuration field ({@code
   * int(input_shape[2]/(self.num_attention_heads - 1))}), so the operand resolution recurses.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dMatmul8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_matmul8.py",
        "einsum_via_matmul",
        2,
        14,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 6)),
            3,
            Set.of(TensorType.of(FLOAT_32, 6, 3, 3))));
  }

  @Test
  public void testDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /**
   * Test for <a href="https://github.com/wala/ML/issues/371">wala/ML#371</a>. A single {@code
   * Dense} layer call inside {@code M.call} with a {@code tf.keras.Input} parameter.
   *
   * <p>Two tensor variables are found: the {@code x} parameter (v3, shape {@code (None, 3)}) and
   * the {@code Dense} result (v25, shape {@code (None, 4)}). Both are correct source-level tensors
   * under a single trampoline context.
   */
  @Test
  public void testDenseModelCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_model_call.py", "M.call", 1, 2, Map.of(3, Set.of(TENSOR_NONE_3_FLOAT32)));
  }

  @Test
  public void testDense2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense2.py", "consume1", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
    test("tf2_test_dense2.py", "consume2", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls at module level, where the second layer's {@code inputs}
   * argument is the return value of the first layer's call. Exercises shape propagation through a
   * layer-call result at script-body scope.
   */
  @Test
  public void testDenseChain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls inside a {@code tf.keras.Model.__call__} method body with
   * direct {@code self.layer1} / {@code self.layer2} attribute reads. Exercises shape propagation
   * through a layer-call result inside a user-defined class method.
   */
  @Test
  public void testDenseChain2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls inside a {@code tf.keras.Model.__call__} method body, where
   * the layers are iterated via a {@code for} loop over a {@code self.layers_list} attribute rather
   * than being accessed by direct attribute name. Exercises shape propagation through a loop-phi'd
   * local whose points-to set spans every list element.
   */
  @Test
  public void testDenseChain3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain3.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }
}
