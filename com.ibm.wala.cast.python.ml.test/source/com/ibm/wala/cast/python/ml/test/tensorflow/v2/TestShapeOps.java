package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_16_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_1_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_5_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_64_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_30_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_512_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.SliceBuiltinOperation;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of shape-manipulation ops ({@code .shape} reads, {@code reshape}, {@code squeeze}, {@code
 * split}, slicing, subscripting, and indexed layer lists), carved from the {@link
 * TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestShapeOps extends AbstractTensorTest {

  /**
   * Regression guard for wala/ML#548: a {@code tf.reshape} on the path to a {@code @tf.function}
   * parameter must not degrade the inferred parameter. Mirrors the cifar10 path in main.py ({@code
   * tf.reshape(labels)} feeding a {@code tf.data.Dataset} whose iteration binds {@code labels});
   * {@code consume(labels)} pins the type. The parameter is concrete and dtype-exact: {@code uint8}
   * (matching cifar10's label dtype), and the union carries both correct batch shapes &mdash;
   * {@code (32,)} for the full batches and {@code (16,)} for the final partial batch (cifar10's
   * 50000 train labels batched by 32 leave {@code 50000 % 32 == 16}, since {@code drop_remainder}
   * defaults to false). So the reshape not only fails to degrade the parameter, the parameter is
   * inferred precisely. See ponder-lab/Input-Signature-Inference-Paper#49.
   */
  @Test
  public void testReshapeToParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_to_param.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_32_UINT8, TENSOR_16_UINT8)));
  }

  /**
   * Pins a slice-subscript result flowing into a dataset (wala/ML#400). Python {@code a_sliced =
   * a[:2, ..., tf.newaxis]; from_tensor_slices((a_sliced, y))} over a {@code (3, 2)} tensor: the
   * subscript is {@code (2, 2, 1)} (slice, ellipsis-fill, newaxis), so each iterated element is
   * {@code (2, 1)}. Because the {@code slice} builtin returns its receiver ({@code
   * Either.forRight(2)}), {@code a_sliced} aliases {@code a}; {@link
   * DatasetFromTensorSlicesGenerator} recovers the slice's shape by dispatching {@link
   * com.ibm.wala.cast.python.ml.client.SliceBuiltinOperation} on the stored value rather than
   * reading the receiver-aliased field PTS.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSliceSubscriptThroughDataset()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_iso_slicesub_ds.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  /**
   * Pins the {@code tf.slice} output shape derived from constant {@code begin}/{@code size}
   * arguments (wala/ML#569). {@code tf.slice(x, [0, 1], [2, 2])} over a {@code (3, 4)} input yields
   * {@code (2, 2)} — all {@code size} entries are non-negative, so the output shape is {@code size}
   * exactly, independent of the input shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Pins the {@code tf.slice} output shape for the "all remaining" case (wala/ML#569). A {@code
   * size[i]} of {@code -1} resolves to {@code input.shape[i] - begin[i]}: {@code tf.slice(x, [1,
   * 0], [-1, 3])} over a {@code (3, 4)} input yields {@code (3 - 1, 3) = (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSliceRemaining()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice.py", "consume_remaining", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output shape of {@code tf.squeeze} with a named axis (wala/ML#513). {@code
   * tf.squeeze(x, [1])} over a {@code (2, 1, 3, 1)} tensor drops only axis 1: {@code (2, 3, 1)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_axis", 1, 1, Map.of(2, Set.of(TENSOR_2_3_1_FLOAT32)));
  }

  /**
   * Pins the output shape of {@code tf.split} with an integer count (wala/ML#717): {@code
   * tf.split(x, 3, 0)} over a {@code (3, 8, 100)} tensor unpacks to pieces of the quotient shape
   * {@code (1, 8, 100)}, NLPGNN's ALBERT entry idiom in miniature.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSplit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_split.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32, asList(new NumericDim(1), new NumericDim(8), new NumericDim(100))))));
  }

  /**
   * The absent-axis default of {@code tf.split} (wala/ML#717): {@code tf.split(x, 2)} over {@code
   * (4, 6)} splits on axis 0, giving pieces of {@code (2, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSplitDefaultAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_split.py",
        "consume_default_axis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6))));
  }

  /**
   * The size-list arm of {@code tf.split} (wala/ML#717): {@code tf.split(x, [1, 3], 0)} produces
   * differently-shaped pieces, which the single-piece model soundly represents with a dynamic
   * dimension at the axis; the other dimension still transfers.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSplitSizeList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_split.py",
        "consume_size_list",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(6))))));
  }

  /**
   * The non-constant-axis guard of {@code tf.split} (wala/ML#717): an opaque {@code axis} leaves
   * the output shape soundly unknown while the dtype still inherits from {@code value}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSplitOpaqueAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_split.py",
        "consume_opaque_axis",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins the output shape of {@code tf.squeeze} with no axis (wala/ML#513). {@code tf.squeeze(x)}
   * over a {@code (2, 1, 3, 1)} tensor drops every statically size-1 axis: {@code (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeAll()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_all", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code tf.squeeze} with a single (non-list) integer axis (wala/ML#513). {@code
   * tf.squeeze(x, 1)} over a {@code (2, 1, 3, 1)} tensor drops axis 1: {@code (2, 3, 1)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeSingleAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_single", 1, 1, Map.of(2, Set.of(TENSOR_2_3_1_FLOAT32)));
  }

  /**
   * Pins {@code tf.squeeze} with multiple named axes (wala/ML#513). {@code tf.squeeze(x, [1, 3])}
   * over a {@code (2, 1, 3, 1)} tensor drops both size-1 axes: {@code (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeMultiAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_multi", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output shape of a non-zero-start slice on the first axis of a multi-dim subscript
   * (wala/ML#406). {@code x[1:3, :, :]} over a {@code (4, 5, 6)} tensor keeps 2 rows and the
   * trailing axes intact: {@code (2, 5, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptMultidimRows()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_multidim.py",
        "consume_rows",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_5_6_FLOAT32)));
  }

  /**
   * Pins the output shape of a middle-axis slice in a multi-dim subscript (wala/ML#406). {@code
   * x[:, 1:, :]} over a {@code (4, 5, 6)} tensor drops the leading element of the middle axis:
   * {@code (4, 4, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptMultidimCols()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_multidim.py",
        "consume_cols",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_6_FLOAT32)));
  }

  /**
   * Pins the output shape of an integer index on the middle axis of a multi-dim subscript
   * (wala/ML#406). {@code x[:, 0, :]} over a {@code (4, 5, 6)} tensor drops the middle axis
   * entirely: {@code (4, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptMultidimIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_multidim.py",
        "consume_index",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_6_FLOAT32)));
  }

  /**
   * Pins the output shape of a multi-dim subscript that mixes a slice, an ellipsis, and a newaxis
   * (wala/ML#406). {@code a[:2, ..., tf.newaxis]} over a {@code (3, 2)} tensor slices the first
   * axis to 2, lets the ellipsis fill the remaining axis (2), and appends a size-1 axis for the
   * newaxis: {@code (2, 2, 1)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptNewaxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_subscript_newaxis.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_1_FLOAT32)));
  }

  @Test
  public void testSliceOpaque()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice_opaque.py", "consume", 0, 0);
  }

  @Test
  public void testSliceOpaqueIter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice_opaque_iter.py", "consume", 0, 0);
  }

  /**
   * Probe for the indexed sub-layer dispatch shape of <a
   * href="https://github.com/wala/ML/issues/661">wala/ML#661</a>: a plain list of sublayers
   * populated by {@code append} in {@code build}, dispatched through a dynamic subscript ({@code
   * self.sub_layers[i](out, training)}) in {@code call} — the miniature of NLPGNN's {@code
   * GAAELayer.encoder}. The inner layer's {@code call} must exist in the call graph with its {@code
   * inputs} parameter tensor-typed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerListCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_list.py", "Inner.call", 1, 1, Map.of(3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Constant-index variant of {@link #testIndexedLayerListCall()} (wala/ML#661): {@code
   * self.sub_layers[0](out, training)} over an append-built list. Pins that the fix does not depend
   * on the subscript being dynamic.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerListCallConst()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_list_const.py", "Inner.call", 1, 1, Map.of(3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * List-literal variant of {@link #testIndexedLayerListCall()} (wala/ML#661): the sublayer list is
   * built as a literal instead of by {@code append}, so the subscript read resolves through the
   * ordinary numeric field. This passed before the fix and pins the discriminator that localized
   * the gap to the append-contents property.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerListCallLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_list_lit.py", "Inner.call", 1, 1, Map.of(3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Indexed dispatch over a comprehension-built sublayer list (wala/ML#661 shape 3, wala/ML#694):
   * {@code self.sub_layers = [Inner() for _ in range(n)]} dispatched through {@code
   * self.sub_layers[i](out)} in a loop. {@code Inner.call} returns a distinctly-shaped {@code (6,
   * 1)} tensor, so a working dispatch would flow {@code (6, 1)} to {@code consume}. The analysis
   * instead reports the pre-loop input's {@code (2, 3)} (carried by the loop phi): the
   * comprehension-built indexed call materializes no callee, so the sub-layer forward result never
   * reaches the sink.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/694">wala/ML#694</a>): once the
   * comprehension-built indexed dispatch materializes its callee, tighten the assertion to the
   * precise {@code (6, 1)} shape (the Python runtime shape).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testIndexedComprehensionLayerListCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // TODO(wala/ML#694): observed-but-imprecise (2, 3); the runtime shape is (6, 1).
    test("tf2_test_layer_list_compr.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Probes wala/ML#661's indexed sub-layer call shape ({@code self.container[i](x)}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_indexed_layer_call.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/581">wala/ML#581</a>: a {@code
   * tf.reshape} whose dim is arithmetic over instance attributes ({@code self.heads *
   * self.out_features}) infers the precise {@code (4, 512)}. Both shape-argument extraction paths
   * now fold the binary op over constant-valued field reads through the points-to analysis (the
   * shared {@link com.ibm.wala.cast.python.ml.types.TensorType#foldArithmeticDim}): the
   * generator-side {@code getShapesFromShapeArgument} and the interpreter-based {@code
   * TensorType.shapeArg}, which previously degraded to {@code DynamicDim} / {@code SymbolicDim}
   * respectively since {@code interpretAsInt} cannot evaluate {@code self.X} as source text.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReshapeSelfArith()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_self_arith.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_512_FLOAT32)));
  }

  /**
   * Coverage companion to {@link #testReshapeSelfArith()} for <a
   * href="https://github.com/wala/ML/issues/581">wala/ML#581</a>: a {@code tf.reshape} dim of
   * {@code self.base + 4} infers the precise {@code (2, 64)}. Exercises the {@code ADD} operator
   * and a literal operand of the arithmetic fold (the sibling fixture covers {@code MUL} over two
   * field reads), so {@link com.ibm.wala.cast.python.ml.types.TensorType#resolveConstantInt}
   * resolves one operand via the symbol table ({@code 4}) and the other via the points-to analysis
   * ({@code self.base}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReshapeSelfArithAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_self_arith_add.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_64_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape2() throws ClassHierarchyException, CancelException, IOException {
    TensorType expectedType =
        new TensorType(
            FLOAT_32,
            asList(
                new SymbolicDim("?"), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

    test("tf2_test_reshape2.py", "f", 1, 1, Map.of(2, Set.of(expectedType)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_28_28_1_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_28_28_1_FLOAT32)));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/741">wala/ML#741</a>: a reshape target mixing
   * an {@code Unresolved} leading element with the literal {@code -1} placeholder surfaces the
   * placeholder as the symbolic unknown-size dimension rather than a fixed size of {@code -1}, so
   * the follow-on broadcast against a fully concrete operand composes instead of flooring to ⊤ on
   * the impossible {@code -1} extent. The markers dominate the concrete sides in the broadcast
   * join, so the sum carries the reshape's {@code (Unresolved, ?, 8)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReshapeMixedPlaceholder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_mixed_placeholder.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, new SymbolicDim("?"), new NumericDim(8))))));
  }

  /**
   * Regression guard for {@code tf.reshape(x, tf.shape(y))} shape inference. Runtime answer is
   * {@code (2, 3)} of {@code float32}. Post wala/ML#538's graceful-degradation fix in {@link
   * com.ibm.wala.cast.python.ml.client.Reshape} (mirroring {@link
   * com.ibm.wala.cast.python.ml.client.BroadcastTo}'s localized try/catch), the analysis no longer
   * throws on the {@code tf.shape(...)} shape arg. The inferred parameter type for {@code x} (vn=2)
   * is currently the imprecise {@code (⊤) of float32} (the opaque-shape-operand unknown-rank pin,
   * wala/ML#703) rather than the precise {@code (2, 3) float32}.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/473">wala/ML#473</a>): tighten the parameter
   * type to {@link #TENSOR_2_3_FLOAT32} when the helper learns to resolve {@code tf.shape(y)}'s
   * shape leaves precisely.
   */
  @Test
  public void testReshapeRuntimeShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_runtime_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Shape-vector provenance (wala/ML#703): {@code tf.reshape(x, t.shape.as_list()[-2:])} resolves
   * to the source tensor's trailing sub-shape. The shape argument's points-to set is empty (the
   * {@code shape} member, {@code as_list}, and the slice are unmodeled in the heap), so {@code
   * Reshape} recovers it by def-use provenance: the slice of the {@code as_list()} of the {@code
   * .shape} of {@code t} of shape {@code (4, 5, 6)}, sliced with {@code [-2:]}, is {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAsListSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_as_list_slice.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Unsliced companion of {@link #testShapeAsListSlice()} (wala/ML#703): {@code tf.reshape(x,
   * t.shape.as_list())} resolves to the source tensor's full shape.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAsList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_as_list.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_5_6_FLOAT32)));
  }

  /**
   * Variable-bound companion of {@link #testShapeAsListSlice()} (wala/ML#703, wala/ML#704): the
   * slice bound is a negated local ({@code shape[-k:]} with a constant {@code k}), mirroring
   * NLPGNN's {@code einsum_via_matmul} idiom ({@code input_shape[-num_inner_dims:]}). The unary
   * negation is constant-folded by the slice-bound resolver, so the trailing sub-shape resolves to
   * the precise {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAsListSliceVar()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_as_list_slice_var.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Interprocedural shape-vector provenance (wala/ML#706): the shape list is produced by a user
   * helper ({@code def get_shape(t): return t.shape.as_list()}, the BERT/ALBERT {@code
   * get_shape_list} pattern), so the def-use walk follows the helper invoke to its returned {@code
   * .shape.as_list()} chain; the callee parameter's interprocedural points-to set resolves the
   * source tensor. The {@code [-2:]} slice of {@code (4, 5, 6)} is the precise {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeHelperSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_helper_slice.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Combines {@link #testShapeHelperSlice()}'s interprocedural hop with {@link
   * #testShapeAsListSliceVar()}'s negated variable bound: {@code get_shape(t)[-k:]} with a constant
   * {@code k} is structurally NLPGNN's {@code einsum_via_matmul} shape read ({@code
   * get_shape_list(input_tensor)[-num_inner_dims:]}). See wala/ML#706 and wala/ML#704.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeHelperSliceVar()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_helper_slice_var.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * The real BERT/ALBERT {@code get_shape_list} vendored verbatim from NLPGNN's {@code
   * nlpgnn/tools.py} (wala/ML#706): unlike {@link #testShapeHelperSliceVar()}'s simplified
   * single-return helper, it patches {@code None} entries after the {@code as_list()} read
   * (subscript writes on the returned list), returns on two paths, and takes default parameters.
   * The def-use walk follows the helper invoke to its {@code .shape.as_list()} chain; the source
   * tensor's static {@code (4, 5, 6)} has no {@code None} axes, so the unpatched chain is the
   * runtime value and the {@code [-k:]} slice is the precise {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeHelperFull()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_helper_full.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Parameter-rooted shape-vector provenance (wala/ML#706): the BERT matrix-reshape round trip
   * ({@code reshape_to_matrix}/{@code reshape_from_matrix}, vendored from NLPGNN's {@code
   * nlpgnn/tools.py}). {@code reshape_from_matrix}'s reshape target is {@code orig_dims + [width]},
   * where {@code orig_dims} slices the {@code orig_shape_list} <em>parameter</em> — the def-use
   * walk roots at a parameter and maps it back to the corresponding argument at the caller's invoke
   * (the {@code get_shape_list(t)} chain), continuing in the caller's frame. The reshape resolves
   * to the precise runtime {@code (4, 5, 6)}; the {@code (20, 6)} member is the rank-2 early-return
   * arm's path-insensitive phantom (runtime skips it since {@code len(input_shape) == 3}), its
   * {@code -1} target folded against the input's element count.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeHelperParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_helper_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 20, 6), TENSOR_4_5_6_FLOAT32)));
  }

  /**
   * {@code np.prod} over a shape-derived list (wala/ML#707): {@code [np.prod(get_shape(t)[-2:])]}
   * folds the product of the static trailing dimensions ({@code 5 * 6}) into the reshape target, so
   * the result is the precise {@code (30,)}. Mirrors NLPGNN's {@code einsum_via_matmul} ({@code
   * inner_dim = np.prod(input_shape[-num_inner_dims:])}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeProd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_prod.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_FLOAT32)));
  }

  /**
   * Distilled guard for the {@code tf.shape} arm of the shape-vector walk (wala/ML#722): a reshape
   * target built from {@code tf.shape(x)[i]} extractions classifies each element by {@code x}'s own
   * axis {@code i} — the {@code None} batch axis contributes its {@link DynamicDim} (the dynamic
   * evidence the wala/ML#721 degradation over-captured as {@link UnresolvedDim}), and the
   * statically known sequence axis folds to its constant, as TensorFlow itself does over static
   * axes.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReshapeTfShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_tf_shape.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(4), new NumericDim(3))))));
  }

  /**
   * Evidence-free twin of {@link #testReshapeTfShape()} (wala/ML#722): when the input's leading
   * axes are fixed runtime sizes the analysis cannot compute, {@code tf.shape(x)[i]} carries no
   * {@code None}-evidence and the reshape target's leading elements classify {@link UnresolvedDim}
   * per the wala/ML#721 criterion, with the trailing constant exact. The pair pins the arm's
   * discriminating variable: the input's own axis evidence, not the target's syntactic pattern.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReshapeTfShapeUnresolved()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_tf_shape_unresolved.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(10))))));
  }

  /**
   * Fold-granularity guard for wala/ML#748: {@code f}'s parameter carries a two-member shape union
   * (the concrete {@code (2, 3, 8)} constant and the dynamic-batch Keras input) through a
   * conditional's φ, so the {@code tf.shape(x)[i]} extractions fold against a multi-member source.
   * Each member keeps its own axis classification &mdash; the concrete-sourced member folds to its
   * constants, the dynamic-sourced one to {@link DynamicDim} on the batch axis &mdash; instead of
   * the ambiguity joining every member to the wider marker.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReshapeTfShapeUnion()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_tf_shape_union.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3, 8),
                new TensorType(
                    FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(3), new NumericDim(8))))));
  }

  /**
   * A configuration attribute as a literal concat element (wala/ML#712): the reshape target
   * concatenates a shape-vector slice with {@code [self.units]}, whose stored value is the
   * constructor argument, exercising the constant fallback of the attribute chase.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeConcatAttr()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_concat_attr.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 6))));
  }

  /**
   * A reshape whose target is a shape-vector slice with an opaque bound (wala/ML#711): the walk
   * cannot resolve {@code k}, so the output shape is soundly unknown, but the input's {@code
   * float32} dtype must survive rather than degrading to full ⊤.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReshapeOpaqueBound()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_opaque_bound.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Single-member counterpart of {@link #testEmbeddingDynamicSize()} (wala/ML#717): dimension
   * arithmetic over a plain (non-φ) shape-vector subscript with a config-sourced factor exercises
   * the singleton fold's degradation, so that element's value is dynamic while the rank and the
   * literal element survive.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReshapeDynamicFactor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_dynamic_factor.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(6))))));
  }

  /**
   * Guard companion of {@link #testShapeProd()} (wala/ML#707): {@code np.prod} with an extra
   * argument ({@code axis=0}) can change the result's rank, so the fold refuses it and the shape
   * position degrades to a dynamic dimension in the walk-side contexts. The interpreter path
   * evaluates the call concretely and contributes the precise {@code (30,)} in its context, so the
   * union carries both.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeProdAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_prod_axis.py",
        "f",
        1,
        1,
        Map.of(
            2,
            Set.of(TENSOR_30_FLOAT32, new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE)))));
  }

  /**
   * Concatenation of two shape vectors (wala/ML#708): {@code get_shape(t)[:1] + get_shape(t)[-2:]}
   * over {@code (4, 5, 6)} composes the reshape target {@code (4, 5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_concat.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_5_6_FLOAT32)));
  }

  /**
   * Strided companion of {@link #testShapeAsListSlice()} (wala/ML#703, wala/ML#709): a constant
   * positive step over a shape list ({@code [::2]}) strides the resolved dimensions, composing the
   * precise {@code (4, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceStep()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_slice_step.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_6_FLOAT32)));
  }

  /**
   * Negative-step companion of {@link #testShapeSliceStep()} (wala/ML#709): a constant negative
   * step reverses and strides the shape list under Python's adjusted-indices semantics ({@code
   * [::-2]} of {@code (4, 5, 6)} composes {@code (6, 4)}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceNegativeStep()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_slice_step_negative.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 6, 4))));
  }

  /**
   * Guard companion of {@link #testShapeAsListSlice()} (wala/ML#703, wala/ML#710): the slice bound
   * is a φ of two constants, so within one context its points-to set is ambiguous and the bound
   * resolver must not assert either slicing alone. The candidate bounds enumerate and the result
   * unions the slicings under each ({@code {(1, 30), (30)}}), which the set-carrying type lattice
   * expresses soundly and more precisely than the ⊤ the ambiguity previously forced.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceAmbiguous()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_slice_ambiguous.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 30), TensorType.of(FLOAT_32, 30))));
  }

  /**
   * Degradation arms of the candidate-bound enumeration (wala/ML#710): a combination count past the
   * cap, a zero step candidate, a non-numeric constant candidate, more candidates on a single bound
   * than the cap, and a non-constant bound in each of the three positions all keep the sound ⊤
   * fallback rather than asserting any particular slicing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceAmbiguousDegrade()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_slice_ambiguous2.py",
        "top",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Precision arms of the candidate-bound enumeration (wala/ML#710): a propagated {@code None}
   * alongside a numeric candidate unions both slicings ({@code (4, 5, 1)} and {@code (4, 5)}), and
   * numerically equal {@code int} and {@code float} candidates deduplicate to one slicing ({@code
   * (5,)}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceAmbiguousPrecise()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_slice_ambiguous2.py",
        "precise",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 4, 5, 1),
                TensorType.of(FLOAT_32, 4, 5),
                TensorType.of(FLOAT_32, 5))));
  }

  /**
   * Guards slice-receiver dtype recovery through a chained slice (<a
   * href="https://github.com/wala/ML/issues/602">wala/ML#602</a>): {@code x_train[:5][:3]} on a
   * {@code (60000, 28, 28) uint8} ndarray yields a {@code (3, 28, 28) uint8} tensor. The outer
   * slice's receiver is the inner slice's result, whose dtype the PTS walk can't see; {@link
   * com.ibm.wala.cast.python.ml.client.TensorGenerator#dtypesFromSSAChain} recovers it by recursing
   * through the dtype-preserving slice op rather than falling back to {@code DType.UNKNOWN}.
   */
  @Test
  public void testSliceChainedDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_slice_chained_dtype.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_28_28_UINT8)));
  }

  /**
   * Guards slice-receiver dtype recovery through a {@code reshape(pad(x))} chain (<a
   * href="https://github.com/wala/ML/issues/602">wala/ML#602</a>), the MRE distilled from
   * MusicTransformer's {@code RelativeGlobalAttention._skewing}. The slice receiver is {@code
   * tf.reshape} of {@code tf.pad}; neither op is itself dtype-modeled ({@code tf.pad} is unmodeled
   * entirely), so the receiver dtype lands at ⊤ unless {@link
   * com.ibm.wala.cast.python.ml.client.TensorGenerator#dtypesFromSSAChain} recurses through the
   * dtype-preserving chain to the concrete {@code float32} input.
   *
   * <p>TODO: the shape stays {@code (1, 1, 3, 2)} (the reshape result) rather than {@code (1, 1, 2,
   * 2)} &mdash; the {@code [:, :, 1:, :]} subscript isn't reducing axis 2, a shape-precision gap
   * tracked by <a href="https://github.com/wala/ML/issues/607">wala/ML#607</a>. This test pins the
   * dtype recovery and the currently-observed shape.
   */
  @Test
  public void testSliceReshapePadDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_slice_reshape_pad_dtype.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_1_3_2_FLOAT32)));
  }

  /**
   * Guards constant-step subscript-slice shape propagation on ndarrays (wala/ML#405): {@code
   * x_train[:5]} on a {@code (60000, 28, 28) uint8} ndarray yields a {@code (5, 28, 28) uint8}
   * tensor. Implemented via {@link SliceBuiltinOperation}; the receiver-shape leak that previously
   * forced this suppression is closed by the set-shape edge-transfer pin on subscript-result
   * variables in {@link PythonTensorAnalysisEngine}.
   */
  @Test
  public void testSubscriptSlicePreservesShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_slice_preserves_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_5_28_28_UINT8)));
  }

  /**
   * Distilled regression guard for the scalar-expression broadcast (wala/ML#718): a tensor scaled
   * by a statically opaque scalar expression ({@code 1.0 / math.sqrt(4.0)}) keeps its shape, since
   * the expression's scalarness is structural even though its value never resolves. The analysis
   * previously erased the result to ⊥. Mirrors NLPGNN's attention-logit scaling.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testScalarExpressionScale()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_scalar_scale.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * Distilled guard for the dimension-provenance split (wala/ML#721): a configuration-sourced size
   * — here an environment read the analysis cannot fold — types as {@link UnresolvedDim}, a fixed
   * runtime size of unknown value, not {@link DynamicDim}, which is reserved for axes the runtime
   * {@code TensorShape} reports as {@code None}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnresolvedDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unresolved_dim.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(100))))));
  }

  /**
   * Pins the fold taint of the dimension-provenance split (wala/ML#721): {@code np.prod} over a
   * shape list whose leading axis is {@code None} stays {@link DynamicDim} — arithmetic over a
   * {@code None} axis is itself {@code None} at run time — rather than degrading to {@link
   * UnresolvedDim}. The runtime guards the {@code None} away before folding (the BERT {@code
   * get_shape_list} idiom), but the static walk sees the unguarded shape. The {@code (24)} member
   * is the guarded arm — the runtime-true fold over the patched {@code [1, 4, 6]} — flowing
   * alongside per the guard-φ.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testProdOverDynamic()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_prod_over_dynamic.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE)),
                TensorType.of(FLOAT_32, 24))));
  }
}
