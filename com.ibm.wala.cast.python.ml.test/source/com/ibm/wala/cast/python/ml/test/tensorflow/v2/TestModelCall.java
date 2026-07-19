package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.MNIST_INPUT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_20_10_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_20_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_20_28_28_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_20_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_20_64_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_20_7_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_10_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of Keras model and layer call dataflow ({@code Model*}/{@code Neural*}/{@code Build*}),
 * carved from the {@link TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestModelCall extends AbstractTensorTest {

  /**
   * Six tensor variables in {@code SequentialModel.__call__}: the {@code x} parameter (vn=3, shape
   * {@code (20, 28, 28) f32}) plus five intermediate SSA values produced by the {@code
   * self.flatten(x) → 100× Dense(64) → self.dropout(x) → self.dense_2(x)} chain. The {@code
   * Flatten} result is concrete {@code (20, 784)} via {@link
   * com.ibm.wala.cast.python.ml.client.FlattenCall} (vn=9); the loop's {@code Dense(64)} output is
   * concrete {@code (20, 64)} (vn=22); the loop-head phi (vn=17) and the {@code Dropout} output
   * (vn=26) union both reachable shapes {@code {(20, 784), (20, 64)}} (the pre-loop entry shape
   * survives because an empty {@code my_layers} would leave {@code x} at the {@code Flatten}
   * shape); the final {@code Dense(10)} output is concrete {@code (20, 10)} (vn=30).
   *
   * <p>The loop's {@code Dense(64)} narrows because {@code range(n)} now returns an iterable
   * (non-empty) list, so the {@code self.my_layers} comprehension populates the list and the {@code
   * self.my_layers[idx]} subscript read resolves to its {@code Dense(64)} elements (<a
   * href="https://github.com/wala/ML/issues/599">wala/ML#599</a>). The direct {@code Dense(10)}
   * narrows via the SSA-chain fallback in {@link
   * com.ibm.wala.cast.python.ml.client.DenseCall#getDefaultShapes} (<a
   * href="https://github.com/wala/ML/issues/358">wala/ML#358</a>).
   */
  @Test
  public void testModelCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call.py",
        "SequentialModel.__call__",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call2.py",
        "SequentialModel.call",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call3.py",
        "SequentialModel.call",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call4.py",
        "SequentialModel.__call__",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  /**
   * Test call string imprecision as described in
   * https://github.com/wala/WALA/discussions/1417#discussioncomment-10085680. This should fail due
   * to https://github.com/wala/ML/issues/207.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#207 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModelCall5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj66/src/tf2_test_model_call5b.py",
          "proj66/tf2_test_model_call5.py",
          "proj66/tf2_test_model_call5a.py"
        },
        "tf2_test_model_call5.py",
        "SequentialModel.__call__",
        "proj66",
        1,
        1,
        Map.of(3, Set.of(MNIST_INPUT)));

    test(
        new String[] {
          "proj66/src/tf2_test_model_call5b.py",
          "proj66/tf2_test_model_call5.py",
          "proj66/tf2_test_model_call5a.py"
        },
        "tf2_test_model_call5a.py",
        "SequentialModel.__call__",
        "proj66",
        1,
        1,
        Map.of(3, Set.of(MNIST_INPUT)));
  }

  /**
   * Test https://github.com/wala/ML/issues/267.
   *
   * <p>Explicit dtype.
   */
  @Test
  public void testModelCall6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call6.py",
        "SequentialModel.__call__",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_INT32)));
  }

  /**
   * Multi-Model separation on the {@code model(x)} call-output path: two distinct Models
   * constructed inside a {@code make_and_call(units, x)} helper that also performs the call. Each
   * user-side {@code make_and_call(...)} returns the model's output, with disjoint shapes (5 vs 7)
   * per call. Call strings alone collapsed both models into one context for {@code
   * Model.__call__}'s output allocation, unioning {@code {(20, 5), (20, 7)}} at every sink; the
   * receiver-keyed trampoline contexts of <a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a> keep each model's dispatch chain
   * separate, so each sink now sees exactly its own model's output shape — same mechanism as {@link
   * #testModelAttributesMultiModelWrapped()}, on the call-output path instead of the
   * trainable-weights path.
   */
  @Test
  public void testModelCallMultiModelWrapped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_call_multi_wrapped.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_20_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testModelCallMultiModelWrapped()} — same fixture, second sink, pinned to
   * the second model's output shape only (wala/ML#679).
   */
  @Test
  public void testModelCallMultiModelWrapped2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_call_multi_wrapped.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_20_7_FLOAT32)));
  }

  /**
   * The explicit {@code model.build(input_shape=...)} contract seed (wala/ML#717), NLPGNN's ALBERT
   * entry idiom in miniature: the runtime inputs are opaque, so the declared {@code (3, 8, 100)}
   * contract seeds the {@code call} input, and the {@code split}/{@code squeeze}/{@code cast} chain
   * carries it to the returned piece: {@code (8, 100) int32}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBuildContract()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_contract.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 8, 100))));
  }

  /**
   * Parameter {@code x} of {@code NeuralNet.call} receives tensors from <b>four</b> source-level
   * call sites of {@code neural_net}, with three distinct runtime shapes (all {@code float32},
   * verified by Python {@code assert} statements in {@code neural_network.py}):
   *
   * <ul>
   *   <li>line 136 ({@code neural_net(x, ...)} inside {@code run_optimization}) &mdash; {@code
   *       (256, 784)} via {@code batch_x} forwarded through {@code x};
   *   <li>line 167 ({@code neural_net(batch_x, ...)} in the training loop) &mdash; {@code (256,
   *       784)};
   *   <li>line 189 ({@code neural_net(x_test, ...)}) &mdash; {@code (10000, 784)};
   *   <li>line 207 ({@code neural_net(test_images)} in the visualization block) &mdash; {@code (5,
   *       784)} via {@code x_test[:n_images]}.
   * </ul>
   *
   * <p>The aspirational expected set for value 3 is therefore the union {@code {(256, 784), (10000,
   * 784), (5, 784) float32}}. A downstream {@code @tf.function(input_signature=...)} consumer would
   * merge these into a single {@code tf.TensorSpec(shape=(None, 784), dtype=tf.float32)} using a
   * wildcard for the varying first dimension, so the union &mdash; not any individual shape &mdash;
   * is the correct source-level specification.
   *
   * <p>Value 3 is currently partially resolved. Shape inference through {@code np.array(x_train,
   * np.float32).reshape([-1, 784]) / 255.0} recovers a partial {@code (?, 784)} (the {@code -1}
   * slot stays symbolic when the receiver's shape is implicit-PK), and the batched shape {@code
   * (256, 784)} follows from {@code from_tensor_slices}'s per-index slice + {@code .batch(256)}.
   * Under the now-union-across-contexts helper, the aggregated actual for vn=3 is {@code {(?, 784)
   * float32, (256, 784) float32}} &mdash; one element per flow family, roughly one per trampoline
   * context. The test fails on types because the concrete test-set shape {@code (10000, 784)} and
   * the visualization slice shape {@code (5, 784)} never fall out of the reshape-{@code -1} slot:
   * the {@code (?, 784)} actual is a coarse approximation that covers both call sites. The
   * remaining analyzer gaps are (a) resolve {@code -1} in {@code reshape([-1, 784])} against the
   * source's concrete mnist-test shape {@code (10000, 28, 28)} to yield {@code (10000, 784)}, and
   * (b) propagate the constant-step slice {@code x_test[:n_images]} so {@code (5, 784)} falls out
   * of {@code (10000, 784)}.
   *
   * <p>Rule-based tensor variable count is 5 (1 parameter {@code x} + 4 intermediate ops {@code
   * fc1}, {@code fc2}, {@code out}, {@code softmax}). With the fix for wala/ML#358, the full {@code
   * fc1 → fc2 → out → softmax} chain narrows along the {@code units} axis ({@code 128 → 256 → 10 →
   * 10}) and every intermediate is registered as a tensor variable. Counts are source-level &mdash;
   * one per distinct value number, deduplicated across calling contexts (wala/ML#371, Option 2)
   * &mdash; so the depth-4 per-caller analysis (train vs. test vs. visualization, wala/ML#379,
   * wala/ML#530) no longer multiplies the count by the number of contexts. The source-level total
   * is 6: parameter {@code x} plus the five registered intermediates of the narrowed chain (one
   * more than the rule-based 4, an extra SSA temporary in the {@code Dense} chain).
   */
  @Test
  public void testNeuralNetwork()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "NeuralNet.call",
        1,
        6,
        Map.of(3, Set.of(TENSOR_256_784_FLOAT32, TENSOR_10000_784_FLOAT32, TENSOR_5_784_FLOAT32)));
  }

  /**
   * {@code cross_entropy_loss(x, y)} receives logits {@code x} (value 2) and labels {@code y}
   * (value 3). At runtime, {@code x} has shape {@code (256, 10)} dtype {@code float32} and {@code
   * y} has shape {@code (256,)} dtype {@code uint8} (verified by Python assert statements in {@code
   * neural_network.py}).
   *
   * <p>Value 2 flows from {@code pred = neural_net(batch_x, is_training=True)}, which dispatches
   * through {@code Model.__call__} into user-defined {@code NeuralNet.call}. After the fix for
   * wala/ML#358 (chained {@code Dense} shape propagation), value 2 is tracked as a tensor parameter
   * with shape {@code (256, 10) float32} &mdash; the final {@code Dense(num_classes=10)} in the
   * chain narrows to {@code (256, 10)} and that shape flows back through the caller chain.
   *
   * <p>The rule-based tensor variable count is 5 (2 parameters {@code x}, {@code y} + 3
   * intermediate ops {@code cast-to-int64}, {@code sparse_softmax_cross_entropy_with_logits},
   * {@code reduce_mean}). Counts are source-level &mdash; one per distinct value number,
   * deduplicated across calling contexts (wala/ML#371, Option 2) &mdash; so the count is 5, the
   * exact rule-based total, rather than the context-multiplied 10 (value 3 ({@code y}) reached
   * three contexts with a dtype that varies per context; that variation is now captured on the type
   * axis, which unions per vn across contexts, not on the count). The former count-axis duplication
   * tracked by wala/ML#388 is subsumed by this source-level counting.
   */
  @Test
  public void testNeuralNetwork2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "cross_entropy_loss",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_10_FLOAT32), 3, Set.of(TENSOR_256_UINT8)));
  }

  /**
   * {@code run_optimization(x, y)} is called with {@code batch_x} and {@code batch_y} from the
   * dataset iteration chain. At runtime, {@code x} has shape {@code (256, 784)} dtype {@code
   * float32} and {@code y} has shape {@code (256,)} dtype {@code uint8} (verified by Python assert
   * statements in {@code neural_network.py}).
   *
   * <p>The test currently fails on value 2's shape only: actual {@code {? of float32}} vs. expected
   * {@code {(256, 784) of float32}}. Dtype routing for slot 0 of the tuple is correct (same state
   * as {@link #testNeuralNetwork()} &mdash; the labels-swap symptom originally reported under
   * wala/ML#396 appears to have been resolved by the per-index delegation work on this branch).
   * What remains is the same shape-propagation gap as {@link #testNeuralNetwork()}: the {@code
   * x_train} chain does not yield a concrete per-index shape through {@code from_tensor_slices} by
   * the time {@code batch_x} reaches {@code run_optimization}'s {@code x}.
   *
   * <p>Rule-based tensor variable count is 6 (2 parameters {@code x}, {@code y} + 4 intermediate
   * ops {@code pred}, {@code loss}, {@code trainable_variables}, {@code gradients}). With the fix
   * for wala/ML#358, {@code pred = neural_net(x, is_training=True)} is now tracked at {@code (256,
   * 10) float32}, bringing the registered count to 4. After wala/ML#430's {@code Gradient}
   * generator, {@code gradients} now registers as one fresh tensor variable (the generator
   * allocates fresh per call rather than aliasing {@code sources}), lifting the count from 4 to 5.
   * {@code trainable_variables} remains a list of tensors and still doesn't register; the residual
   * gap from 5 to 6 is tracked by wala/ML#391.
   */
  @Test
  public void testNeuralNetwork3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "run_optimization",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_784_FLOAT32), 3, Set.of(TENSOR_256_UINT8)));
  }

  /**
   * {@code accuracy(y_pred, y_true)} is called from two sites: the training loop with {@code
   * accuracy(pred, batch_y)} where {@code pred} has shape {@code (256, 10)} dtype {@code float32}
   * and {@code batch_y} has shape {@code (256,)} dtype {@code uint8}; and the test-set evaluation
   * with {@code accuracy(pred, y_test)} where {@code pred} has shape {@code (10000, 10)} dtype
   * {@code float32} and {@code y_test} has shape {@code (10000,)} dtype {@code uint8} (verified by
   * Python assert statements in {@code neural_network.py}). The static analysis should union these
   * types for each parameter.
   *
   * <p>Rule-based tensor variable count is 7 (2 parameters {@code y_pred}, {@code y_true} + 5
   * intermediate ops {@code argmax}, {@code cast-to-int64}, {@code equal}, {@code cast-to-float32},
   * {@code reduce_mean}). Counts are source-level &mdash; one per distinct value number,
   * deduplicated across calling contexts (wala/ML#371, Option 2) &mdash; so the count is 7, the
   * exact rule-based total: all five intermediates now register ({@code argmax} via the {@code
   * ReadDataFallback} of wala/ML#437; {@code tf.equal}/{@code tf.cast} no longer drop out under
   * wala/ML#386/wala/ML#387), with the depth-4 per-caller contexts (wala/ML#379, wala/ML#530)
   * deduplicated rather than multiplying the count.
   *
   * <p>Value 2 ({@code y_pred}) is tracked as a tensor parameter after the fix for wala/ML#358
   * (chained {@code Dense} shape propagation): the final {@code Dense(num_classes=10)} in {@code
   * NeuralNet.call} narrows to {@code (256, 10)} and that shape propagates back into {@code
   * accuracy}'s {@code y_pred} parameter. This test runs at k-CFA depth 4 (wala/ML#379) so {@code
   * NeuralNet.call} is analyzed per caller and its layer-output ({@code pred}) no longer collapses
   * the training shape into the test context (wala/ML#530); value 2 is therefore the per-context
   * union {@code {(256, 10) float32, ? float32}}. The test-context contribution is ⊤ shape because
   * the {@code x_test} chain resolves to a rank-preserving but shape-unknown tensor by the time it
   * reaches {@code NeuralNet.call}'s first {@code Dense} operand; closing that gap would narrow the
   * ⊤ to {@code (10000, 10)} (orthogonal to #358/#530).
   */
  @Test
  public void testNeuralNetwork4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Source-level count is 7 (wala/ML#371, Option 2): the 2 parameters plus the 5
    // intermediate ops, deduplicated across the depth-4 calling contexts. `accuracy()`'s
    // `tf.argmax(...)` is a legitimate tensor source per #380 and now registers via
    // `ReadDataFallback` (#437); under (CGNode, vn) counting its per-context call sites
    // inflated this to 14. Parameter-type expectations (`y_pred`, `y_true`) unchanged.
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "accuracy",
        2,
        7,
        Map.of(
            2,
            // Per-context union: training call site `accuracy(pred, batch_y)` gives `(256, 10)`;
            // the test-set call site `accuracy(pred, y_test)` resolves to ⊤ shape because the
            // `x_test` chain is shape-unknown by the time it reaches `NeuralNet.call`'s first
            // `Dense` operand. With the depth-4 context separation (wala/ML#530), `argmax`'s result
            // no longer leaks the training shape into the test context. TODO: the test-context ⊤
            // should narrow to `(10000, 10)` once the `x_test` shape gap is closed.
            Set.of(TENSOR_256_10_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32),
            3,
            Set.of(TENSOR_256_UINT8, TENSOR_10000_UINT8)));
  }

  /**
   * Regression guard for wala/ML#657: a {@code tf.keras.Model} subclass reached via {@code
   * model(x)} callable dispatch keeps its call-graph node even when another module defines a
   * same-named {@code class Model}. The subclass uses the ubiquitous bare-import idiom {@code from
   * tensorflow.keras import Model; class MyModel(Model)}; when a second module ({@code
   * tf2_657_collide.py}) also defines {@code class Model}, the bare base name previously
   * mis-resolved across modules, so {@code MyModel}'s superclass came back {@code null} and {@code
   * MyModel.call} was dropped from the call graph (which is the empty {@code getNodes(...)} that
   * fails Hybridize's side-effect, recursion, and primitive-parameter preconditions). The fix falls
   * back to {@code object} when the base name has no local resolution, matching the collision-free
   * case.
   */
  @Test
  public void testModelSubclassNameCollision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(emptyList(), new String[] {"tf2_657_model_call.py", "tf2_657_collide.py"});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    assertTrue(
        "MyModel.call should have a call-graph node despite a colliding `class Model` in another"
            + " module (wala/ML#657).",
        CG.stream()
            .anyMatch(
                n ->
                    n.getMethod()
                        .getSignature()
                        .contains("tf2_657_model_call.py.MyModel.call.do")));
  }

  /**
   * Probe for the model-forward tuple-of-reshapes shape: a {@code tf.keras.Model} subclass whose
   * {@code call} returns a tuple of {@code tf.reshape} results, unpacked at the top-level call site
   * and passed to {@code consume}. Discriminates the reshape-producer axis against the passing
   * layer-tuple-return shape ({@link #testCollectionProbeLayerTupleReturn()}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelForwardTupleReshapeReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_tuple_reshape_return.py",
        "consume",
        2,
        2,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32), 3, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Control for {@link #testModelForwardTupleReshapeReturn()}: identical shape but the returned
   * tuple's elements are elementwise results rather than {@code tf.reshape} results.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelForwardTupleAddReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_tuple_add_return.py",
        "consume",
        2,
        2,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32), 3, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the generator-fed model-forward shape: as {@link
   * #testModelForwardTupleReshapeReturn()} but the model input arrives via {@code next()} on a
   * generator function, tuple-unpacked at the call site. The generator transit drops the tensor
   * typing (<a href="https://github.com/wala/ML/issues/696">wala/ML#696</a>), so {@code consume}
   * previously saw zero tensor parameters; a regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelForwardTupleReshapeGenInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_tuple_reshape_gen_input.py",
        "consume",
        2,
        2,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32), 3, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Attribute-read arm of the {@code build} stored-attribute resolution (wala/ML#725): the stored
   * value is an attribute read off a same-body holder object whose attribute holds an {@code
   * input_shape} subscript (an empty-points-to-set chain), so the subscript resolver must classify
   * the non-numeric string member as an attribute read rather than an index. The chase is depth-one
   * by construction, so the holder's attribute does not resolve and the weight's leading dimension
   * soundly degrades.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredAttrRead()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_attr_read",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  /**
   * Numeric-string subscript arm of the {@code build} stored-attribute resolution (wala/ML#725):
   * the stored value is a dict subscript whose key is the numeric string {@code "2"} and whose held
   * value is an {@code input_shape} subscript (an empty-points-to-set chain), so the subscript
   * resolver's string-index parse succeeds and the dict is rejected as a shape vector (its rank
   * never resolves). The depth-one chase then leaves the attribute unresolved and the weight's
   * leading dimension soundly degrades.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredDictKey()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_dict_key",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  /**
   * Nested-class config idiom for the {@code build} stored-attribute resolution (wala/ML#725): a
   * class declared in the method body writes its member as a field-put on the class object's local,
   * so the receiver-class write scan must recognize the put's non-{@code self} receiver and skip it
   * rather than misattributing it; the depth-one chase then leaves the attribute unresolved and the
   * weight's leading dimension soundly degrades.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredNestedCfg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_nested_cfg",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  /**
   * Direct companion of {@link #testBuildStoredNestedCfg()} (wala/ML#725): the nested config
   * class's member is read directly in the weight-shape literal, with no intermediate stored
   * attribute, exercising the read-side handling of the class-object member access.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredDirectNestedCfg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_direct_nested_cfg",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  /**
   * Unresolvable-index arm of the {@code build} stored-attribute resolution (wala/ML#725): the
   * subscript index is a runtime int the analysis cannot resolve, so the constant-index sentinel
   * guard rejects the subscript and the weight's leading dimension degrades.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredOpaqueIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_opaque_index",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  /**
   * Out-of-bounds subscript arm of the {@code build} stored-attribute resolution (wala/ML#725): the
   * first write's index is beyond the resolved rank (a runtime-guarded {@code try}/{@code except}
   * read), so the bounds check rejects it and the unresolvable write makes the attribute
   * unresolvable, degrading the weight's leading dimension.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredOobIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_oob_index",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  /**
   * Negative out-of-bounds companion of {@link #testBuildStoredOobIndex()} (wala/ML#725): the
   * normalized index falls below zero.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuildStoredNegativeOobIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_build_stored_attr.py",
        "apply_negative_oob_index",
        2,
        3,
        Map.of(
            2,
            Set.of(TensorType.of(FLOAT_32, 2, 6)),
            3,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(5))))));
  }

  @Test
  public void testModelInit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_init.py", "check_positional", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_keyword", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_mixed", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_subclass", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test(
        "tf2_test_model_init.py",
        "check_multiple",
        2,
        2,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Pins the output type of a functional {@code tf.keras.Model} call whose output shape <em>differs
   * from</em> its input shape: a {@code Dense(3)} model maps a {@code (1, 2)} call input to a
   * {@code (1, 3)} output. {@code ModelCall} recovers the model's output generator (the {@code
   * Dense(3)} call, reached via the {@code outputs} construction argument) and reports the
   * transformed {@code (1, 3)} shape — both for positional ({@code Model(in, out)}) and keyword
   * ({@code Model(outputs=...)}) construction. Before wala/ML#537, {@code ModelCall} fell back to
   * the call's input shape when the output generator wasn't reached, which would have reported the
   * unsound {@code (1, 2)} here (input shape, not output). Companion to {@link #testModelInit}
   * (whose {@code Dense(2)}-on-dim-2 models are shape-preserving, so they can't distinguish the two
   * behaviors) and to {@link #testGanTutorialGeneratorLoss} (whose deep convolutional chain leaves
   * the output generator shapeless, exercising the ⊤ path).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelCallOutputShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_output_shape.py",
        "consume_positional",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
    test(
        "tf2_test_model_call_output_shape.py",
        "consume_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
  }

  /**
   * Regression guard for wala/ML#358.
   *
   * <p>Derived from {@link #testModelCall()} (see {@code tf2_test_model_call.py}) by adding a
   * {@code consume(x)} call inside {@code SequentialModel.__call__} immediately after {@code x =
   * self.dense_2(x)}. At that point {@code x} has shape {@code (20, 10)} and dtype {@code float32}:
   * the chain traces {@code (20, 28, 28)} input → {@code Flatten} → {@code (20, 784)} → 100× {@code
   * Dense(64)} → {@code Dropout} → {@code Dense(10)} → {@code (20, 10)}. {@link
   * com.ibm.wala.cast.python.ml.client.DenseCall#getDefaultShapes} recovers the input shape via an
   * SSA-chain fallback when the PTS walk's allocating-node dispatch can't identify the upstream
   * layer call (Flatten, Dropout, or another Dense).
   */
  @Test
  public void testModelCallConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_consume.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_20_10_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/599">wala/ML#599</a>.
   *
   * <p>Derived from {@link #testModelCall()} (see {@code tf2_test_model_call.py}) by adding a
   * {@code consume(x)} call <em>inside</em> the {@code for layer in self.my_layers} loop,
   * immediately after {@code x = layer(x)}. At that point {@code x} is the loop's {@code Dense(64)}
   * output, shape {@code (20, 64)} and dtype {@code float32}.
   *
   * <p>This pins the loop-iterated layer call's output, which is the gap wala/ML#599 closes:
   * because {@code range(n)} now returns an iterable (non-empty) list, the {@code self.my_layers}
   * comprehension populates the list, the {@code self.my_layers[idx]} subscript read resolves to
   * its {@code Dense(64)} elements, and the loop call's output narrows to {@code (20, 64)} rather
   * than carrying the upstream {@code Flatten} shape {@code (20, 784)} unchanged. {@link
   * com.ibm.wala.cast.python.ml.client.DenseCall} breaks the resulting input-shape self-recursion
   * (the loop's collapsed 1-CFA node feeds its own output back in) via a per-thread cycle guard.
   */
  @Test
  public void testModelCallLoopConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_loop_consume.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_20_64_FLOAT32)));
  }
}
