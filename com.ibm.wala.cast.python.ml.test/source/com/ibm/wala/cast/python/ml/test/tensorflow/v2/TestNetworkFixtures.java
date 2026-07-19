package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SPARSE_TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_100_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_256_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_64_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_10_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_1_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_8_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_96_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Collections.emptyMap;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.SparseTensorType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * End-to-end network-fixture tests (the GAN/autoencoder/MLP/logistic-regression tutorials, the
 * LSTM/Transformer blocks, the GNN family, and the CRF metrics), carved from the {@link
 * TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestNetworkFixtures extends AbstractTensorTest {

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   *
   * <p>{@code summarize_weights(step)} takes only an {@code int} step counter (not a tensor, hence
   * parameter count 0). Inside the function, dict lookups {@code weights[w]} and {@code biases[b]}
   * feed 4 tensor variables to {@code tf.summary.histogram}. Previously this test registered 5: the
   * extra was a spurious {@code v2} (the {@code step} parameter) typed {@code {(256, 784) float32}}
   * via PA-graph propagation from the {@code enumerate()} call at the caller ({@code for step, (x,
   * y) in enumerate(ds, 1):}). The leak was at the PA substrate &mdash; tensor types flowing to the
   * enumerate tuple's field-0 slot through the PTS-graph edge, not at generator dispatch.
   *
   * <p>Closed by wala/ML#409: {@link com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine}
   * now detects enumerate-first-field reads structurally and pins their {@link
   * com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis} state to empty via a {@code DropOp}
   * edge transfer that both clears existing leaked state and FIXes the slot against further
   * predecessor updates. Count correctly reports 4.
   */
  @Test
  public void testTensorboardExample()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tensorboard_example.py", "summarize_weights", 0, 4);
  }

  /**
   * {@code train_step(images, generator, discriminator, ...)} receives {@code image_batch} from the
   * training loop inside {@code train}. {@code image_batch} comes from iterating a dataset built
   * from mnist data via {@code train_images[..., None].astype(np.float32)} and {@code
   * from_tensor_slices(...).shuffle(...).batch(256)}. At runtime {@code image_batch} has shape in
   * {@code {(256, 28, 28, 1), (96, 28, 28, 1)}} dtype {@code float32} (60000 training images / 256
   * = 234 full batches + 1 partial batch of 96; verified by Python assert statements in {@code
   * tensorflow_gan_tutorial.py}).
   *
   * <p>Expected tensor variable count: 7. After wala/ML#430's {@code Gradient} generator allocates
   * a fresh tensor per {@code tape.gradient(...)} call instead of aliasing {@code sources}, each of
   * the two gradient calls in {@code train_step} (one for the generator, one for the discriminator)
   * registers an additional local tensor variable, lifting the count from the prior master baseline
   * of 5 to 7. Value 2 ({@code images}) is inferred concretely as {@code {(256, 28, 28, 1), (96,
   * 28, 28, 1)} float32}: the mnist pipeline resolves end to end. {@code mnist.load_data()}
   * supplies {@code (60000, 28, 28)} (<a
   * href="https://github.com/wala/ML/issues/361">wala/ML#361</a>), {@code [..., None]} and the
   * {@code (x - 127.5) / 127.5} binop chain carry it to {@code (60000, 28, 28, 1)}, {@code
   * from_tensor_slices} takes the element shape {@code (28, 28, 1)}, and {@code .batch(256)}
   * produces the two batch shapes (<a
   * href="https://github.com/wala/ML/issues/356">wala/ML#356</a>).
   */
  @Test
  public void testGanTutorial()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "train_step",
        1,
        7,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * Same structure as {@link #testGanTutorial()} but with {@code @tf.function} applied to {@code
   * train_step}. Runtime types for {@code image_batch} are identical and are verified via the
   * Python assert statements in {@code tensorflow_gan_tutorial.py} (not duplicated in {@code
   * tensorflow_gan_tutorial2.py} since the two files are structurally identical apart from the
   * decorator): shape in {@code {(256, 28, 28, 1), (96, 28, 28, 1)}}, dtype {@code float32}.
   * Expected count 7, same accounting as {@link #testGanTutorial()}: the two `tape.gradient(...)`
   * calls each contribute one fresh tensor variable post-wala/ML#430 (5 to 7). Value 2 is inferred
   * concretely as in {@link #testGanTutorial()}; the mnist binop-chain pipeline resolves (<a
   * href="https://github.com/wala/ML/issues/356">wala/ML#356</a>, <a
   * href="https://github.com/wala/ML/issues/361">wala/ML#361</a>).
   */
  @Test
  public void testGanTutorial2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial2.py",
        "train_step",
        1,
        7,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * Pins {@code generator_loss(fake_output)}'s parameter type. {@code fake_output} flows from
   * {@code discriminator(generated_images, ...)}; runtime shape is {@code (batch, 1) float32} since
   * the discriminator ends with a {@code Dense(1)} layer.
   *
   * <p>Inferred as {@code float32} with shape {@code ⊤} (unknown). The dtype is concrete and sound.
   * The shape is unknown rather than wrong: previously {@code ModelCall.getDefaultShapes} fell back
   * to the call's <em>input</em> shape when no output generator resolved, emitting the unsound
   * {@code (None, 100)} ({@code 100} is the generator's noise input dim from {@code
   * tf.keras.Input((100,))}, not the discriminator's {@code Dense(1)} output dim). A {@code
   * tf.keras.Model} generally transforms its input shape, so that fallback is removed
   * (wala/ML#537): the input shape now only refines a recovered output shape's batch dim, never
   * substitutes for it.
   *
   * <p>The runtime shape is {@code (256, 1)}/{@code (96, 1)} (verified by running the fixture:
   * {@code noise (256, 100)} -&gt; generator {@code (256, 28, 28, 1)} -&gt; discriminator {@code
   * (256, 1)}), confirming the old {@code (None, 100)} was the noise shape leaking through, not the
   * discriminator output.
   *
   * <p>TODO: tighten to {@code {(256, 1) float32, (96, 1) float32}}. The output node <em>is</em>
   * reachable now (the {@code outputs} construction argument points at the {@code Dense(1)} call),
   * but that {@code DenseCall} returns {@code null} shapes here because its input — the {@code
   * Flatten} of a {@code Conv2D} chain — isn't shape-tracked; recovering {@code (batch, 1)} needs
   * {@code DenseCall} output-shape inference through chained layer calls (tracked by <a
   * href="https://github.com/wala/ML/issues/358">wala/ML#358</a>), which in turn needs {@code
   * Conv2D}/{@code Flatten} output-shape modeling. Concrete batch dims 256/96 come from {@code
   * train_step}'s {@code images} parameter, per {@link #testGanTutorial}. (wala/ML#537 fixed the
   * unsound mis-propagation; this residual precision is wala/ML#358's domain.)
   */
  @Test
  public void testGanTutorialGeneratorLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "generator_loss",
        1,
        2,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code multilayer_perceptron(x)}'s parameter type. Function body mirrors {@code
   * multilayer_perceptron} from {@code
   * YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py}. Uses raw {@code
   * tf.matmul} / {@code tf.add} / {@code tf.nn.sigmoid} / {@code tf.nn.softmax} against global
   * {@code tf.Variable} weights and biases — a different pattern from the {@code Dense}-layer
   * subclass-Model approach already covered by {@code testNeuralNetwork*}.
   *
   * <p>{@code x} is inferred as {@code (100, 784) float32}, flowing from the caller's {@code
   * batch_x = tf.constant(np.ones((100, 784), dtype=np.float32))}. This relies on the {@code numpy
   * → tf.constant} dtype/shape bridge fixed in <a
   * href="https://github.com/wala/ML/issues/539">wala/ML#539</a> (see {@link
   * #testConstantFromNumpy} for the isolated guard).
   *
   * <p>The local-tensor count is 16 (up from 14 before wala/ML#539): with {@code x} now typed, two
   * further {@code tf.matmul}/{@code tf.add} intermediates that consume it are recognized as
   * tensors.
   */
  @Test
  public void testMultilayerPerceptron()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_multilayer_perceptron.py",
        "multilayer_perceptron",
        1,
        16,
        Map.of(2, Set.of(TENSOR_100_784_FLOAT32)));
  }

  /**
   * Pins {@code logistic_regression(x)}'s parameter type. Function body mirrors {@code
   * logistic_regression} from {@code
   * aymericdamien/TensorFlow-Examples/.../2_BasicModels/logistic_regression.py}, a real-world
   * image-classification utility (logistic regression: {@code softmax(W x + b)} over global {@code
   * tf.Variable} weights and biases), for tensor-type inference coverage. Like {@link
   * #testMultilayerPerceptron}, it uses raw {@code tf.matmul} / {@code tf.nn.softmax} rather than
   * the {@code Dense}-layer subclass-{@code Model} approach of {@code testNeuralNetwork*}.
   *
   * <p>{@code x} is inferred as {@code (100, 784) float32}—both shape and dtype concrete—flowing
   * from the caller's {@code tf.constant(np.ones((100, 784), dtype=np.float32))} via the {@code
   * numpy→tf.constant} bridge (<a href="https://github.com/wala/ML/issues/539">wala/ML#539</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLogisticRegression()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_logistic_regression.py",
        "logistic_regression",
        1,
        6,
        Map.of(2, Set.of(TENSOR_100_784_FLOAT32)));
  }

  /**
   * Pins {@code nce_loss(x_embed, y)}'s parameter types. Function body mirrors {@code nce_loss}
   * from {@code aymericdamien/TensorFlow-Examples/.../2_BasicModels/word2vec.py}, a real-world
   * word-embedding utility (the averaged noise-contrastive-estimation loss over global {@code
   * tf.Variable} embedding/weight/bias matrices), for tensor-type inference coverage.
   *
   * <p>Both parameters are inferred concretely—shape and dtype: {@code x_embed} as {@code (4, 10)
   * float32} and {@code y} as {@code (4, 1) int32}, flowing from the {@code
   * tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNceLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nce_loss.py",
        "nce_loss",
        2,
        5,
        Map.of(2, Set.of(TENSOR_4_10_FLOAT32), 3, Set.of(TENSOR_4_1_INT32)));
  }

  /**
   * Pins {@code evaluate(x_embed)}'s parameter type. Function body mirrors {@code evaluate} from
   * {@code aymericdamien/TensorFlow-Examples/.../2_BasicModels/word2vec.py}, a real-world
   * word-embedding utility (the cosine similarity between an input embedding and every row of the
   * global embedding matrix), for tensor-type inference coverage.
   *
   * <p>{@code x_embed} is inferred as {@code (4, 10) float32}—both shape and dtype concrete—flowing
   * from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEvaluate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_evaluate.py", "evaluate", 1, 13, Map.of(2, Set.of(TENSOR_4_10_FLOAT32)));
  }

  /**
   * Pins {@code random_jitter(input_image, real_image)}'s parameter types. Function body (and the
   * {@code resize}/{@code random_crop} helpers it calls) mirrors {@code random_jitter} from {@code
   * YunYang1994/TensorFlow2.0-Examples/.../Pix2Pix.py}, a real-world image-to-image translation
   * utility (random resize/crop/mirror data augmentation), for tensor-type inference coverage.
   *
   * <p>Both image parameters are inferred concretely—shape and dtype—as {@code (256, 256, 3)
   * float32}, flowing from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRandomJitter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_random_jitter.py",
        "random_jitter",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_256_3_FLOAT32), 3, Set.of(TENSOR_256_256_3_FLOAT32)));
  }

  /**
   * Pins {@code MaskSparseCategoricalCrossentropy.__call__(y_true, y_predict, input_mask)}'s
   * parameter types. Class and method body mirror {@code MaskSparseCategoricalCrossentropy} from
   * {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/Losess.py}, a real-world NLP utility (a mask-weighted
   * sparse-categorical-crossentropy loss), for tensor-type inference coverage. Unlike the {@code
   * tf.keras.Model.call} layer-chain methods ({@code testNeuralNetwork*}), this is a loss {@code
   * __call__} on a plain class that reduces its inputs to a scalar.
   *
   * <p>All three parameters are inferred concretely—shape and dtype: {@code y_true} as {@code (4,)
   * int32}, {@code y_predict} as {@code (4, 10) float32}, and {@code input_mask} as {@code (4,)
   * float32}, flowing from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMaskedSparseCrossentropy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_masked_sparse_ce.py",
        "MaskSparseCategoricalCrossentropy.__call__",
        3,
        7,
        Map.of(
            3, Set.of(TENSOR_4_INT32),
            4, Set.of(TENSOR_4_10_FLOAT32),
            5, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * Pins {@code LSTM.call(x)}'s parameter type. Class and method body mirror the {@code LSTM}
   * recurrent model from {@code
   * aymericdamien/TensorFlow-Examples/.../3_NeuralNetworks/recurrent_network.py}, a real-world
   * sequence-classification utility (a built-in {@code tf.keras.layers.LSTM} followed by a {@code
   * Dense} read-out), for tensor-type inference coverage.
   *
   * <p>The input parameter {@code x} is recovered concretely on both axes—{@code (256, 28, 28)
   * float32}—flowing from the {@code lstm_net(x, is_training=True)} call site through {@code
   * tf.keras.Model.__call__} dispatch.
   *
   * <p>The forward-pass locals ({@code lstm_layer} output, {@code out} output, {@code softmax}) are
   * inferred as {@code float32} but with <em>unknown shape</em>: the built-in {@code LSTM}/{@code
   * Dense} output shapes are not narrowed (the layer-chain shape gap tracked by <a
   * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>). The dtype axis—the load-bearing
   * one—is exact; only shape is ⊤.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLstmCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_lstm_call.py", "LSTM.call", 1, 4, Map.of(3, Set.of(TENSOR_256_28_28_FLOAT32)));
  }

  /**
   * Pins {@code GCNLayer.call(node_embeddings, adjacency_lists)}'s tensor-parameter type. The
   * {@code GCNLayer} model and the {@code GraphConvolution}/{@code MessagePassing} layers it builds
   * on are vendored verbatim from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/GCN.py}, {@code
   * nlpgnn/gnn/GCNConv.py}, {@code nlpgnn/gnn/messagepassing.py}); only the driver and a
   * reachable-slice {@code nlpgnn/gnn/utils.py} (just the {@code GNNInput} named tuple) are
   * bespoke. This is a real-world graph-neural-network utility (a two-layer graph-convolution
   * message-passing model), exercised for tensor-type inference coverage across a multi-module
   * import chain.
   *
   * <p>The tensor parameter {@code node_embeddings} is recovered concretely on both axes—{@code (4,
   * 8) float32}—flowing from the driver's {@code model(node_embeddings, adjacency_lists,
   * training=False)} call site through {@code tf.keras.Model.__call__} dispatch, across the {@code
   * driver→GCN→GCNConv→MessagePassing} module boundaries. ({@code adjacency_lists} is a Python list
   * of edge tensors, not a tensor itself, so it is not a tensor parameter.)
   *
   * <p>The message-passing <em>output</em> locals (the {@code gc1}/{@code gc2} results) are still
   * inferred as ⊤, but the cause is now downstream of the matmul rather than the {@code NamedTuple}
   * field read. The aggregation ops are modeled, and the {@code GNNInput} {@code NamedTuple} field
   * read {@code node_embeddings = inputs.node_embeddings} recovers {@code (4, 8) float32} (read off
   * the instance field in the heap; <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>), so {@code tf.linalg.matmul} and
   * {@code propagate} inside {@code GraphConvolution.call} now type to {@code float32}. What
   * remains is the forward-result hop: {@code GraphConvolution.call} returns a typed tensor, but
   * that result does not propagate to the caller's {@code self.gc1(...)} local &mdash; the
   * user-subclass forward-result-typing gap (wala/ML#570, akin to <a
   * href="https://github.com/wala/ML/issues/595">wala/ML#595</a>). The decorated function's input
   * signature, the analysis goal, is nonetheless exact. The returned {@code tf.math.softmax} result
   * counts as one more tracked variable since the {@code math.softmax} alias was wired
   * (wala/ML#711).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_proj/nlpgnn/__init__.py",
          "gcn_proj/nlpgnn/gnn/__init__.py",
          "gcn_proj/nlpgnn/gnn/utils.py",
          "gcn_proj/nlpgnn/gnn/messagepassing.py",
          "gcn_proj/nlpgnn/gnn/GCNConv.py",
          "gcn_proj/nlpgnn/models/__init__.py",
          "gcn_proj/nlpgnn/models/GCN.py",
          "gcn_proj/tf2_test_gcn_call.py"
        },
        "nlpgnn/models/GCN.py",
        "GCNLayer.call",
        "gcn_proj",
        1,
        7,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins {@code GATLayer.call(node_embeddings, adjacency_lists)}'s tensor-parameter type. The
   * attention counterpart of {@link #testGcnCall()}: the {@code GATLayer} model and the {@code
   * GraphAttentionConvolution}/{@code MessagePassing} layers it builds on are vendored verbatim
   * from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/GAT.py}, {@code nlpgnn/gnn/GATConv.py},
   * {@code nlpgnn/gnn/messagepassing.py}); only the driver and a reachable-slice {@code
   * nlpgnn/gnn/utils.py} (the {@code GNNInput} named tuple plus {@code masksoftmax}/{@code
   * maybe_num_nodes}) are bespoke.
   *
   * <p>The tensor parameter {@code node_embeddings} is recovered concretely on both axes &mdash;
   * {@code (4, 8) float32} &mdash; flowing from the driver's {@code model(node_embeddings,
   * adjacency_lists, training=False)} call site through {@code tf.keras.Model.__call__} dispatch,
   * across the {@code driver→GAT→GATConv→MessagePassing} module boundaries. ({@code
   * adjacency_lists} is a Python list of edge tensors, not a tensor itself, so it is not a tensor
   * parameter.) As with {@link #testGcnCall()}, the decorated function's input signature &mdash;
   * the analysis goal &mdash; is exact, while the internal layer-output locals stay ⊤. The five
   * tracked tensor variables are {@code node_embeddings} (vn=3) and the {@code dropout1} output
   * (vn=11), both concrete {@code (4, 8) float32} (dropout preserves shape and dtype), the {@code
   * gc1}/{@code gc2} attention-convolution outputs (vn=18, vn=38), both ⊤ on each axis, plus the
   * returned {@code tf.math.softmax} result, typed since the {@code math.softmax} alias was wired
   * (wala/ML#711). The layer outputs are ⊤ for the same reason as GCN: {@code
   * GraphAttentionConvolution.call} unwraps its input from a {@code GNNInput} {@code NamedTuple}
   * field, which is not tracked as a tensor (wala/ML#579), so the input arrives ⊤ and the attention
   * aggregation through {@code tf.math.unsorted_segment_*} (wala/ML#570, wala/ML#582) inherits it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gat_proj/nlpgnn/__init__.py",
          "gat_proj/nlpgnn/gnn/__init__.py",
          "gat_proj/nlpgnn/gnn/utils.py",
          "gat_proj/nlpgnn/gnn/messagepassing.py",
          "gat_proj/nlpgnn/gnn/GATConv.py",
          "gat_proj/nlpgnn/models/__init__.py",
          "gat_proj/nlpgnn/models/GAT.py",
          "gat_proj/tf2_test_gat_call.py"
        },
        "nlpgnn/models/GAT.py",
        "GATLayer.call",
        "gat_proj",
        1,
        5,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Captured-gap reproduction for <a href="https://github.com/wala/ML/issues/659">wala/ML#659</a>:
   * in the vendored GAT subject, {@code maybe_num_nodes}'s {@code index} parameter (vn=2) is
   * currently <em>not</em> typed as a tensor, even though it receives a real tensor.
   *
   * <p>The chain: {@code MessagePassing._calculate_messages_all_type} computes {@code edge_targets
   * = adjanceny_list_edge_type[:, 1]} (a subscript-slice of an {@code enumerate(adjacency_lists)}
   * element), which flows through {@code GATConv.message_function}'s {@code edge_target} into
   * {@code masksoftmax(alpha, edge_target)} and then {@code maybe_num_nodes(index, ...)}. In {@code
   * propagate} the adjacency list types correctly to {@code (E, 2) int32}, but passed into {@code
   * _calculate_messages_all_type} the {@code adjacency_lists} parameter is context-collapsed on the
   * shared message-passing summary (merged with the {@code float32} {@code node_embeddings}),
   * losing its {@code int32}/rank-2 type. The corrupted ⊤ makes the {@code enumerate} element
   * empty-shaped, so the {@code [:, 1]} subscript returns ⊥ and {@code index} never types. This is
   * a regression from 0.52.9 (bisected to the {@code SliceBuiltinOperation} empty-shape change in
   * wala/ML#656, which unmasked the pre-existing collapse; ⊤ before, ⊥ after), and the root cause
   * is the same context-collapse class as <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/659">wala/ML#659</a>): {@code index} should
   * type as a tensor once the shared-summary context collapse is resolved. Flip this to assert vn=2
   * <em>is</em> typed when it is fixed.
   */
  @Test
  public void testGatMaybeNumNodesIndexUntyped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            getPathFiles("gat_proj"),
            new String[] {
              "gat_proj/nlpgnn/__init__.py",
              "gat_proj/nlpgnn/gnn/__init__.py",
              "gat_proj/nlpgnn/gnn/utils.py",
              "gat_proj/nlpgnn/gnn/messagepassing.py",
              "gat_proj/nlpgnn/gnn/GATConv.py",
              "gat_proj/nlpgnn/models/__init__.py",
              "gat_proj/nlpgnn/models/GAT.py",
              "gat_proj/tf2_test_gat_call.py"
            });
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    // Guard against a vacuous pass: the captured-gap assertion below only checks that vn=2 is
    // *absent* from the typed set, which is trivially satisfied if `maybe_num_nodes` is no longer
    // reached at all (e.g. the entrypoint or file list changes). Require a reachable node so the
    // test fails, rather than passing silently, when the reproduction stops exercising the target.
    assertTrue(
        "The reproduction must reach `maybe_num_nodes`; otherwise this captured-gap guard passes"
            + " vacuously (wala/ML#659).",
        CG.stream().anyMatch(n -> n.getMethod().getSignature().contains("maybe_num_nodes")));

    // Collect the parameter value numbers that `maybe_num_nodes` types as tensors. Its `index`
    // parameter is vn=2 (vn=1 is the function object).
    //
    // Coverage note: while the wala/ML#659 gap holds, no `maybe_num_nodes` parameter types, so the
    // `isParameter() && ...maybe_num_nodes` branch is never taken and `typedParamVns.add(...)` is
    // never reached (Codecov reports them as an uncovered line and partial branch). That is the
    // captured gap itself; both become covered when #659 is fixed and the assertion below is
    // flipped
    // to expect vn=2 typed.
    Set<Integer> typedParamVns = new HashSet<>();
    analysis.forEach(
        pt -> {
          if (pt.fst instanceof LocalPointerKey) {
            LocalPointerKey lpk = (LocalPointerKey) pt.fst;
            if (lpk.isParameter()
                && lpk.getNode().getMethod().getSignature().contains("maybe_num_nodes"))
              typedParamVns.add(lpk.getValueNumber());
          }
        });
    assertFalse(
        "Captured gap for wala/ML#659: `maybe_num_nodes`'s `index` (vn=2) is currently untyped"
            + " because `adjacency_lists` is context-collapsed on the shared message-passing"
            + " summary. Flip this assertion when the collapse is resolved.",
        typedParamVns.contains(2));
  }

  /**
   * Pins {@code GCN.call(self, features, adj)}'s tensor-parameter types. The {@code GCN} layer is
   * vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * deep_recommenders/keras/models/retrieval/gcn.py}); only the driver is bespoke. This is a
   * real-world graph-convolution layer, exercised for tensor-type inference coverage of a
   * <em>sparse tensor parameter</em>: unlike the other vendored layer methods, {@code adj} is a
   * {@code tf.SparseTensor}, on which the layer branches ({@code isinstance(adj, tf.SparseTensor)})
   * to use {@code tf.sparse.sparse_dense_matmul}. The driver feeds a sparse {@code adj}, mirroring
   * {@code train_gcn_on_cora_keras.py}'s {@code GCN(32)(feats, g)} call site where {@code g} is a
   * {@code scipy.sparse} adjacency.
   *
   * <p>Both parameters are recovered concretely: {@code features} (vn=3) as {@code (4, 8) float32}
   * and the sparse {@code adj} (vn=4) as {@code (4, 4) float32} with {@link
   * com.ibm.wala.cast.python.ml.types.TensorType.Layout#SPARSE} layout (a {@link
   * SparseTensorType}). So the sparse parameter is typed precisely, including its sparse layout,
   * rather than collapsing to dense or ⊤ &mdash; the sparse {@code TensorType} representation of <a
   * href="https://github.com/wala/ML/issues/588">wala/ML#588</a>. (The {@code **kwargs} parameter
   * is not a tensor and is not extracted.) Emitting {@code tf.SparseTensorSpec} for such a
   * parameter in an inferred signature is the downstream consumer's job; this confirms the typed
   * input it needs is available.
   *
   * <p>The local-tensor count rose from 4 to 7 when the Keras lazy-{@code build} protocol was
   * modeled (<a href="https://github.com/wala/ML/issues/595">wala/ML#595</a>): {@code self._kernel}
   * now dispatches, so {@code outputs} (the {@code Dense} result), the residual {@code outputs +=
   * features} value, and the return alias gained tensor types &mdash; a precision gain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnSparseCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_sparse_proj/deep_recommenders/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/retrieval/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/retrieval/gcn.py",
          "gcn_sparse_proj/tf2_test_gcn_sparse_call.py"
        },
        "deep_recommenders/keras/models/retrieval/gcn.py",
        "GCN.call",
        "gcn_sparse_proj",
        2,
        7,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32), 4, Set.of(SPARSE_TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the chained-layer forward result (<a href="https://github.com/wala/ML/issues/595">
   * wala/ML#595</a>): a value bound to a user-defined Keras {@code Layer}'s call is tensor-typed at
   * the call site, so it flows as a tensor into a downstream function. The fixture chains two
   * {@code GCN} layers (vendored from {@code deep_recommenders}), mirroring {@code
   * train_gcn_on_cora_keras.py}'s {@code x = GCN(32)(feats, g); GCN(num_classes)(x, g)}, and sinks
   * the first layer's output through {@code consume(hidden)}.
   *
   * <p>{@code hidden} (the {@code GCN.call} return, a {@code Dense} output) is tensor-classified
   * because the modeled Keras lazy-{@code build} protocol invokes {@code GCN.build}, giving the
   * {@code build()}-created {@code self._kernel} sublayer a points-to set. With constructor keyword
   * arguments forwarded to {@code __init__} (wala/ML#664) and the layer-method trampolines keyed on
   * the receiver instance (wala/ML#679), each instance's {@code build} constructs its own {@code
   * Dense}, so the runtime-true {@code (4, 16) float32} is inferred without the other instance's
   * spurious {@code (4, 8)}. The remaining {@code ? of float32} member comes from the statically
   * dead {@code outputs += features} residual branch ({@code self._residual} is constantly {@code
   * False}), which the path-insensitive analysis still evaluates.
   *
   * <p>TODO: Expect exactly {@code (4, 16) float32} once <a
   * href="https://github.com/wala/ML/issues/681">wala/ML#681</a> prunes branches guarded by
   * statically-constant instance fields.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnChainConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_chain_proj/deep_recommenders/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/retrieval/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/retrieval/gcn.py",
          "gcn_chain_proj/tf2_test_gcn_chain_call.py"
        },
        "tf2_test_gcn_chain_call.py",
        "consume",
        "gcn_chain_proj",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 16), TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code Transformer.call(encoder_inputs, decoder_inputs)}'s parameter types. The {@code
   * Transformer} layer and the {@code MultiHeadAttention}/{@code ScaledDotProductAttention} layers
   * it builds on are vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * deep_recommenders/keras/models/nlp/transformer.py} and {@code .../multi_head_attention.py});
   * only the driver is bespoke. This is a real-world sequence-to-sequence utility (a
   * multi-head-attention encoder/decoder transformer), exercised for tensor-type inference coverage
   * across a multi-module import chain.
   *
   * <p>Both token-sequence parameters are recovered concretely on both axes—{@code (2, 5) int32}
   * each—flowing from the driver's {@code transformer(encoder_inputs, decoder_inputs)} call site
   * through {@code tf.keras.layers.Layer.__call__} dispatch, across the {@code
   * deep_recommenders→nlp→transformer} package boundaries.
   *
   * <p>With {@code tf.keras.backend} modeled (<a
   * href="https://github.com/wala/ML/issues/666">wala/ML#666</a>), the padding mask {@code masks =
   * K.equal(inputs, 0)} is a third function-local tensor, typed {@code (2, 5)} bool. With {@code
   * add_weight} consuming its arguments (wala/ML#667), {@code embeddings =
   * K.gather(self.embeddings, inputs)} is a fourth: the embedding table types {@code (?, 8)}
   * float32 and the {@code gather} pass-through carries it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransformerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tr_proj/deep_recommenders/__init__.py",
          "tr_proj/deep_recommenders/keras/__init__.py",
          "tr_proj/deep_recommenders/keras/models/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/multi_head_attention.py",
          "tr_proj/deep_recommenders/keras/models/nlp/transformer.py",
          "tr_proj/tf2_test_transformer_call.py"
        },
        "deep_recommenders/keras/models/nlp/transformer.py",
        "Transformer.call",
        "tr_proj",
        2,
        4,
        Map.of(3, Set.of(TENSOR_2_5_INT32), 4, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins {@code crf_unary_score(tag_indices, sequence_lengths, inputs)}'s parameter types. Function
   * body mirrors {@code crf_unary_score} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}, a
   * real-world linear-chain CRF function exercised for tensor-type inference coverage. Its {@code
   * tf.reshape} with runtime-derived dimensions ({@code tf.shape(inputs)[0]}) previously crashed
   * the analysis (wala/ML#567). The local count includes the {@code tf.range(...) * shape-element}
   * offset chain, whose rank-0 co-operands preserve the range results' shapes through the
   * elementwise scalar rule (wala/ML#723).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfUnaryScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_unary_score",
        3,
        22,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_binary_score(tag_indices, sequence_lengths, transition_params)}'s parameter
   * types. Function body mirrors {@code crf_binary_score} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage.
   *
   * <p>The local count includes the flat-index arithmetic ({@code start_tag_indices * num_tags +
   * end_tag_indices}), typed exactly {@code (2, 2) int32}: a {@code tf.shape} element is a rank-0
   * tensor, so the elementwise scalar-co-operand rule preserves the tensor operand's shape through
   * the nested product (wala/ML#723).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfBinaryScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_binary_score",
        3,
        14,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)}'s
   * parameter types. Function body mirrors {@code crf_sequence_score} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfSequenceScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_sequence_score",
        4,
        9,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_3_INT32),
            4, Set.of(TENSOR_2_INT32),
            5, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_log_norm(inputs, sequence_lengths, transition_params)}'s parameter types.
   * Function body mirrors {@code crf_log_norm} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}
   * for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfLogNorm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_log_norm",
        3,
        9,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_forward(inputs, state, transition_params, sequence_lengths)}'s parameter types.
   * Function body mirrors {@code crf_forward} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}
   * for tensor-type inference coverage. {@code crf_forward} is reached only through {@code
   * crf_log_norm} (its sole NLPGNN caller), which passes {@code inputs} and {@code state} from
   * {@code tf.slice}/{@code tf.squeeze} results over a {@code (2, 3, 4)} constant. {@code inputs}
   * (from {@code tf.slice(inputs, [0, 1, 0], [-1, -1, -1])}) infers as {@code (2, 2, 4)} via the
   * {@code begin}/{@code size} shape derivation (wala/ML#569). {@code state} (from {@code
   * tf.squeeze(tf.slice(inputs, [0, 0, 0], [-1, 1, -1]), [1])}) infers as {@code (2, 4)}: the
   * {@code Slice} shape gives {@code (2, 1, 4)} and {@code tf.squeeze}'s axis-1 removal
   * (wala/ML#513) drops the singleton. Both were previously {@code ⊤}-shaped (dtype only,
   * wala/ML#568).
   *
   * <p>The local tensor-variable count is {@code 15}: the two {@code tf.transpose} calls now
   * allocate distinct tensors rather than aliasing their inputs as first-argument {@code
   * pass_through} did (wala/ML#513 bucket 2a), so one additional reassignment is counted as a
   * tensor local. The widened operand walks (wala/ML#739) add one more: {@code sequence_lengths -
   * 1}'s operator result passes the operand-tensor-evidence gate (wala/ML#451) through the {@code
   * tf.cast} chain and types the runtime-true {@code (2,)} int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_forward",
        4,
        17,
        Map.of(
            2, Set.of(TENSOR_2_2_4_FLOAT32),
            3, Set.of(TENSOR_2_4_FLOAT32),
            4, Set.of(TENSOR_4_4_FLOAT32),
            5, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Pins {@code crf_decode_forward(inputs, state, transition_params, sequence_lengths)}'s parameter
   * types. Function body mirrors {@code crf_decode_forward} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage. The caller passes
   * {@code inputs} from {@code x[:, 1:, :]} and {@code state} from {@code x[:, 0, :]}, recovered as
   * {@code (2, 2, 4)} and {@code (2, 4)} respectively via the multi-dim subscript-slice modeling
   * (wala/ML#406): {@code inputs[:, 1:, :]} drops the leading element of the middle axis ({@code 3
   * → 2}) and {@code inputs[:, 0, :]} drops the middle axis entirely (an integer index).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfDecodeForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_decode_forward",
        4,
        6,
        Map.of(
            2, Set.of(TENSOR_2_2_4_FLOAT32),
            3, Set.of(TENSOR_2_4_FLOAT32),
            4, Set.of(TENSOR_4_4_FLOAT32),
            5, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Pins {@code _gather_elements_along_row(data, column_indices)}'s parameter types. Function body
   * mirrors {@code _gather_elements_along_row} from {@code
   * deep_recommenders/keras/models/retrieval/sbcnm.py} (identical to {@code _take_long_axis} in
   * {@code factorized_top_k} per the source), a real-world recommender-systems utility, for
   * tensor-type inference coverage. Both parameters infer concretely: {@code data} as {@code (2, 4)
   * float32} and {@code column_indices} as {@code (2, 3) int32}. (The function's
   * runtime-dimensioned final {@code tf.reshape} leaves the local <em>result</em> symbolic, but the
   * parameters themselves are exact.)
   *
   * <p>The local tensor-variable count is {@code 10}. The {@code tf.shape} shape-vector arm
   * (wala/ML#722) types the final reshape to its exact runtime {@code (2, 3)}, and the {@code
   * tf.range} remodel (wala/ML#723) types the {@code range}/{@code expand_dims} chain soundly
   * ({@code (Unresolved,)} and {@code (Unresolved, 1)}); the tile and the flat-index arithmetic
   * stay at sound unknowns, since a fold over an {@code Unresolved} axis has no numeric value.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatherElementsAlongRow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gather_elements_along_row.py",
        "_gather_elements_along_row",
        2,
        10,
        Map.of(
            2, Set.of(TENSOR_2_4_FLOAT32),
            3, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * {@code encoder(x)} receives {@code x} from call sites {@code decoder(encoder(x))} inside {@code
   * run_optimization} (training loop) and {@code decoder(encoder(batch_x))} at the module-level
   * test loop. Both call sites pass batches of shape {@code (256, 784)} dtype {@code float32}
   * (verified by Python assert statements in {@code autoencoder.py}).
   *
   * <p>Expected tensor variable count: 11 &mdash; the distinct SSA vns in {@code encoder}'s body
   * that get tensor types: the parameter {@code x} (vn=2) plus the full layer-1 and layer-2
   * computation: {@code weights['encoder_h1']} (vn=19), {@code biases['encoder_b1']} (vn=24),
   * layer-1 matmul (vn=15), layer-1 add (vn=11), {@code layer_1} sigmoid (vn=4), and the
   * corresponding layer-2 values {@code weights['encoder_h2']} (vn=41), {@code
   * biases['encoder_b2']} (vn=45), layer-2 matmul (vn=38), layer-2 add (vn=35), {@code layer_2}
   * sigmoid (vn=31, returned). Counts are source-level &mdash; one per distinct vn, deduplicated
   * across the 2 call-site contexts (wala/ML#371, Option 2) &mdash; so the two contexts no longer
   * double the total to 22.
   */
  @Test
  public void testAutoencoder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "encoder", 1, 11, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code mean_square(reconstructed, original)} is called only from {@code run_optimization}
   * (itself a FUT &mdash; {@link #testAutoencoder3()}). Its arguments are {@code
   * reconstructed_image = decoder(encoder(x))} and {@code x}, both of which have runtime shape
   * {@code (256, 784)} dtype {@code float32}. Direct call-site asserts aren't possible (they would
   * perturb {@code run_optimization}'s count), so the runtime types are verified indirectly through
   * the {@code batch_x} asserts at the training-loop call of {@code run_optimization}.
   *
   * <p>Expected tensor variable count: 4 — 2 parameters plus 2 intermediate-op tensors picked up by
   * #196's {@code ReadDataFallback}: {@code original - reconstructed} (vn=9) and {@code tf.pow}
   * (vn=13), both flow-through {@code (256, 784) float32}. {@code tf.reduce_mean}'s scalar result
   * is the function's return and isn't tracked as a separate variable. Per-op generators tracked in
   * #449 would tighten the asserted types from ⊤-shape/UNKNOWN-dtype to concrete shapes without
   * changing the count.
   *
   * <p>Value 2 ({@code reconstructed}) resolves to concrete {@code (256, 784) float32} after the
   * {@code tensorflow/python/ops/variables/Variable} allocatable-class declaration was added in
   * {@code tensorflow.xml} (closes <a
   * href="https://github.com/wala/WALA/issues/1889">wala/WALA#1889</a>). With the variable
   * allocations now registered in the heap model, the {@code matmul → add → sigmoid →
   * user-function-return} shape chain fully resolves end-to-end.
   */
  @Test
  public void testAutoencoder2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "autoencoder.py",
        "mean_square",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_784_FLOAT32), 3, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code run_optimization(x)} is called from the training loop with {@code batch_x} of shape
   * {@code (256, 784)} dtype {@code float32} (verified by Python assert statements in {@code
   * autoencoder.py}).
   *
   * <p>Expected local tensor variable count: 5, the tensor-producing values {@code encoder(x)}
   * result, {@code decoder(...) = reconstructed_image}, {@code mean_square(...) = loss}, {@code
   * gradients} (a fresh tensor variable since wala/ML#430's {@code Gradient} generator), and the
   * returned {@code loss} alias. The count dropped from 6 to 5 with wala/ML#750: {@code
   * trainable_variables = list(weights.values()) + list(biases.values())} is a Python list
   * concatenation, not a tensor, so it no longer falsely registers as a tensor variable.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAutoencoder3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "run_optimization", 1, 5, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code decoder(x)} is called from the same two sites as {@code encoder} ({@link
   * #testAutoencoder()}), but with {@code x} being the output of {@code encoder}. Since {@code
   * encoder}'s layer 2 has dim {@code num_hidden_2 = 64}, {@code decoder} receives {@code (256, 64)
   * float32} (verified by a Python assert in the test loop).
   *
   * <p>Expected tensor variable count: 11 (parallel to {@link #testAutoencoder()}). Same body
   * structure: 11 distinct SSA vns covering the parameter plus the two-layer {@code
   * weights[...]}/{@code biases[...]} / matmul / add / sigmoid chain, counted source-level (one per
   * distinct vn, deduplicated across the 2 call-site contexts; wala/ML#371, Option 2).
   *
   * <p>Value 2 ({@code decoder}'s {@code x} parameter) resolves to concrete {@code (256, 64)
   * float32} after the {@code tensorflow/python/ops/variables/Variable} allocatable-class fix
   * (closes <a href="https://github.com/wala/WALA/issues/1889">wala/WALA#1889</a>). The {@code
   * encoder → decoder} shape chain now flows end-to-end through the XML-summary return keys.
   */
  @Test
  public void testAutoencoder4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "decoder", 1, 11, Map.of(2, Set.of(TENSOR_256_64_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.gather_nd}. Output dtype is inherited from the {@code
   * params} input (here float32); the output shape is {@code indices.shape[:-1] +
   * params.shape[indices.shape[-1]:]}, so a (2, 2) table indexed by (2, 2) depth-2 indices yields
   * (2,). See {@link com.ibm.wala.cast.python.ml.client.GatherNd} (wala/ML#449 Tier 8).
   */
  @Test
  public void testGatherNd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gather_nd.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/669">wala/ML#669</a>: {@code
   * build_model} is vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * examples/train_transformer_on_imdb_keras.py}), a functional {@code tf.keras.Model} whose
   * weight-graph walk ({@code Model.getWeightShapes}) resolves {@code Dense}/{@code MatMul} weight
   * shapes — the exact frames that crashed with {@code IllegalStateException} on 0.52.12 when a
   * WALA 1.8.0 non-constant key reached {@code getConstantValues}. The crash's non-constant {@code
   * units} key itself arises only under the consumer-side speculative call-graph configuration,
   * which this harness does not enable, so this guard pins that the walk completes; the {@code
   * getConstantValues} degrade contract is exercised structurally.
   *
   * <p>With wala/ML#670 fixed, the walk traverses past the head {@code Dense} into the transformer
   * (an unresolvable input no longer stops the trace-back, and {@code GlobalAveragePooling1D} is
   * modeled — see {@link #testGap1dWeights()}), but it still yields no weight shapes here: the
   * pooling input is the vendored transformer's forward output, whose shape is the wala/ML#570
   * residual. TODO: expect the concrete weight-shape union once <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a> is fixed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransformerWeights()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tr_proj/deep_recommenders/__init__.py",
          "tr_proj/deep_recommenders/keras/__init__.py",
          "tr_proj/deep_recommenders/keras/models/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/multi_head_attention.py",
          "tr_proj/deep_recommenders/keras/models/nlp/transformer.py",
          "tr_proj/tf2_test_transformer_weights.py"
        },
        "tf2_test_transformer_weights.py",
        "consume",
        "tr_proj",
        0,
        0,
        emptyMap());
  }

  /**
   * Pins the vendored {@code LayerNormalization} forward result: {@code add_weight}-created {@code
   * gamma}/{@code beta} dispatch (wala/ML#595, wala/ML#618) and the normalization body types.
   * Receiver-keyed contexts (wala/ML#679) dropped the shapeless-and-dtypeless union member, and
   * with parameter defaults materializing in the pointer analysis (wala/ML#743), the {@code
   * epsilon=1e-6} co-operand classifies as a runtime scalar, so the {@code variance + epsilon}
   * broadcast preserves the operand shape and the result is the runtime-true concrete {@code (2, 3,
   * 8)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testVendoredLayerNorm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_ln.py"
        },
        "probe_ln.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 8))));
  }
}
