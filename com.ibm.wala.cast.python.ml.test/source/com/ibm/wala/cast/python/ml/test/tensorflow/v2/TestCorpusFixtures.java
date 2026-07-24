package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.UNKNOWN;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.test.categories.WholeProjectFixtures;
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
import org.junit.experimental.categories.Category;

/**
 * Whole-project corpus-fixture tests (the vendored NLPGNN, gpt-2, MusicTransformer, BiLSTM, and
 * TextCNN subjects), carved as one unit from the {@code TestTensorflow2Model} monolith
 * (wala/ML#635). Kept together deliberately: these are the suite's long pole (the wala/ML#755
 * parallel-split candidate), and their loop-carried expectations are sensitive to JVM test ordering
 * (wala/ML#753). The assertions are verbatim.
 */
@Category(WholeProjectFixtures.class)
public class TestCorpusFixtures extends AbstractTensorTest {

  /**
   * Regression guard for wala/ML#655, the full symptom-A chain in one fixture: the NLPGNN {@code
   * BilstmAttention} model (whose {@code predict} forwards to {@code self(inputs, training)} and
   * whose {@code call} delegates to a user-defined child {@code BiLSTM} layer built from unmodeled
   * sublayers) fed from the {@code TFLoader} {@code FixedLenFeature}/{@code TFRecordDataset}
   * source. The child {@code BiLSTM.call}'s {@code inputs} parameter types to {@code (128,)} int64,
   * flowing the parsed {@code input_ids} field through {@code model.predict(X)} → {@code __call__}
   * → {@code call} → {@code self.bilstm(inputs, training)}. This demonstrates that the symptom-A
   * cases are unit-reproducible via the source pipeline (not the affected file, which carries no
   * call site or source) and that the cause was the source typing, not the {@code __call__}
   * forwarding the issue title hypothesized. Before the {@code FixedLenFeature} fix, {@code inputs}
   * came back non-tensor.
   */
  @Test
  public void testBilstmLoaderE2e()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_bilstm_loader_e2e.py",
        "BiLSTM.call",
        1,
        2,
        Map.of(3, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code BiLSTM.call}
   * row on the vendored subject: the {@code BiLSTM} layer is vendored verbatim from {@code
   * kyzhouhzau/NLPGNN} ({@code nlpgnn/layers/bilstm.py}); only the driver is bespoke. The {@code
   * inputs} parameter (an integer token-ID tensor feeding a Keras {@code Embedding}) recovers
   * {@code (2, 5)} int32 exactly, flowing from the driver's {@code layer(tokens, training=False)}
   * call site through {@code tf.keras.layers.Layer.__call__} dispatch.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBilstmCallVendored()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "bilstm_proj/nlpgnn/__init__.py",
          "bilstm_proj/nlpgnn/layers/__init__.py",
          "bilstm_proj/nlpgnn/layers/bilstm.py",
          "bilstm_proj/tf2_test_bilstm_call.py"
        },
        "nlpgnn/layers/bilstm.py",
        "BiLSTM.call",
        "bilstm_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code
   * TextCNN.predict} row on the vendored subject ({@code kyzhouhzau/NLPGNN}, {@code
   * nlpgnn/models/TextCNN.py}, same vendoring as {@link #testTextcnnCall()}): {@code predict}
   * forwards {@code inputs} to the model through {@code self(inputs, training)}. The {@code inputs}
   * parameter recovers {@code (2, 5)} int32 exactly; the second tracked local is the forward
   * result, float32 with ⊤ shape (the chained-layer body, wala/ML#358/wala/ML#530).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTextcnnPredict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "textcnn_proj/nlpgnn/__init__.py",
          "textcnn_proj/nlpgnn/models/__init__.py",
          "textcnn_proj/nlpgnn/models/TextCNN.py",
          "textcnn_proj/tf2_test_textcnn_predict.py"
        },
        "nlpgnn/models/TextCNN.py",
        "TextCNN.predict",
        "textcnn_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Whole-project-layout probe for <a href="https://github.com/wala/ML/issues/678">wala/ML#678</a>:
   * the subject's structure in miniature — nested entry scripts ({@code tests/TG/EN/}) each
   * defining a same-named {@code GenGPT2} over a root-level {@code nlpgnn} package (inner {@code
   * gpt2.GPT2} model, shared closure-dispatching {@code samples.sample_sequence}) — the layout the
   * fixture-scale reproductions lacked. Both siblings keep their call-graph nodes and type — the
   * layout is excluded as the wala/ML#678 trigger — with the wala/ML#685 cross-sibling closure
   * union as the pinned imprecision.
   *
   * <p>TODO: Expect each sibling's own shape once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnSliceGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nlpgnn_slice_proj/nlpgnn/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/callbacks.py",
          "nlpgnn_slice_proj/nlpgnn/models/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/models/gpt2.py",
          "nlpgnn_slice_proj/nlpgnn/sample/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/sample/samples.py",
          "nlpgnn_slice_proj/tests/__init__.py",
          "nlpgnn_slice_proj/tests/TG/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/generation.py",
          "nlpgnn_slice_proj/tests/TG/EN/interactive.py"
        },
        "tests/TG/EN/generation.py",
        "GenGPT2.predict",
        "nlpgnn_slice_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testNlpgnnSliceGeneration()} (wala/ML#678) — the {@code interactive}
   * entry script, the one degraded in the whole-project consumer runs.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnSliceInteractive()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nlpgnn_slice_proj/nlpgnn/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/callbacks.py",
          "nlpgnn_slice_proj/nlpgnn/models/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/models/gpt2.py",
          "nlpgnn_slice_proj/nlpgnn/sample/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/sample/samples.py",
          "nlpgnn_slice_proj/tests/__init__.py",
          "nlpgnn_slice_proj/tests/TG/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/generation.py",
          "nlpgnn_slice_proj/tests/TG/EN/interactive.py"
        },
        "tests/TG/EN/interactive.py",
        "GenGPT2.predict",
        "nlpgnn_slice_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * The complete NLPGNN subject vendored verbatim under {@code nlpgnn_full_proj} (wala/ML#690);
   * shared by {@link #testNlpgnnFullGeneration()} and {@link #testNlpgnnFullInteractive()} so the
   * two sibling guards cannot diverge as the fixture changes. Package-visible so {@link
   * TestNlpgnnTransformer} analyzes the same vendored project without duplicating the file list.
   */
  static final String[] NLPGNN_FULL_PROJECT_FILES = {
    "nlpgnn_full_proj/nlpgnn/__init__.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/GCNConvv0.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/__init__.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/scatter.py",
    "nlpgnn_full_proj/nlpgnn/bpemd/bpe.py",
    "nlpgnn_full_proj/nlpgnn/callbacks.py",
    "nlpgnn_full_proj/nlpgnn/datas/__init__.py",
    "nlpgnn_full_proj/nlpgnn/datas/checkpoint.py",
    "nlpgnn_full_proj/nlpgnn/datas/dataloader.py",
    "nlpgnn_full_proj/nlpgnn/datas/graphloader.py",
    "nlpgnn_full_proj/nlpgnn/datas/word2vec.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GAAEConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GATConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GINConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GSConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/RGCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/TGCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/__init__.py",
    "nlpgnn_full_proj/nlpgnn/gnn/glob.py",
    "nlpgnn_full_proj/nlpgnn/gnn/messagepassing.py",
    "nlpgnn_full_proj/nlpgnn/gnn/utils.py",
    "nlpgnn_full_proj/nlpgnn/layers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/layers/albert_transformer.py",
    "nlpgnn_full_proj/nlpgnn/layers/attention.py",
    "nlpgnn_full_proj/nlpgnn/layers/bilstm.py",
    "nlpgnn_full_proj/nlpgnn/layers/decoder.py",
    "nlpgnn_full_proj/nlpgnn/layers/dense.py",
    "nlpgnn_full_proj/nlpgnn/layers/embedding.py",
    "nlpgnn_full_proj/nlpgnn/layers/gpt2_transformer.py",
    "nlpgnn_full_proj/nlpgnn/layers/normalization.py",
    "nlpgnn_full_proj/nlpgnn/layers/transformer.py",
    "nlpgnn_full_proj/nlpgnn/metrics/Losess.py",
    "nlpgnn_full_proj/nlpgnn/metrics/Metric.py",
    "nlpgnn_full_proj/nlpgnn/metrics/__init__.py",
    "nlpgnn_full_proj/nlpgnn/metrics/crf.py",
    "nlpgnn_full_proj/nlpgnn/metrics/type.py",
    "nlpgnn_full_proj/nlpgnn/models/GAAE.py",
    "nlpgnn_full_proj/nlpgnn/models/GAT.py",
    "nlpgnn_full_proj/nlpgnn/models/GCN.py",
    "nlpgnn_full_proj/nlpgnn/models/GIN.py",
    "nlpgnn_full_proj/nlpgnn/models/GraphSage.py",
    "nlpgnn_full_proj/nlpgnn/models/PCNN.py",
    "nlpgnn_full_proj/nlpgnn/models/RGCN.py",
    "nlpgnn_full_proj/nlpgnn/models/TextCNN.py",
    "nlpgnn_full_proj/nlpgnn/models/TextGCN2019.py",
    "nlpgnn_full_proj/nlpgnn/models/__init__.py",
    "nlpgnn_full_proj/nlpgnn/models/albert.py",
    "nlpgnn_full_proj/nlpgnn/models/bert.py",
    "nlpgnn_full_proj/nlpgnn/models/gpt2.py",
    "nlpgnn_full_proj/nlpgnn/models/tucker.py",
    "nlpgnn_full_proj/nlpgnn/optimizers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/optimizers/optim.py",
    "nlpgnn_full_proj/nlpgnn/sample/__init__.py",
    "nlpgnn_full_proj/nlpgnn/sample/samples.py",
    "nlpgnn_full_proj/nlpgnn/savers.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/gpt2_tokenization.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/tokenization.py",
    "nlpgnn_full_proj/nlpgnn/tools.py",
    "nlpgnn_full_proj/setup.py",
    "nlpgnn_full_proj/tests/CLS/ALBERT/albert_cls_test.py",
    "nlpgnn_full_proj/tests/CLS/ALBERT/albert_cls_train.py",
    "nlpgnn_full_proj/tests/CLS/BERT/bert_classification_test.py",
    "nlpgnn_full_proj/tests/CLS/BERT/bert_classification_train.py",
    "nlpgnn_full_proj/tests/CLS/BilstmAttention/bilstm_attention_test.py",
    "nlpgnn_full_proj/tests/CLS/BilstmAttention/bilstm_attention_train.py",
    "nlpgnn_full_proj/tests/CLS/TextCNN/text_cnn_test.py",
    "nlpgnn_full_proj/tests/CLS/TextCNN/text_cnn_train.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/attention.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/bert.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/build_graph.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/train_text_gcn.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/transformer.py",
    "nlpgnn_full_proj/tests/GNN/auto_encoder/GAAE.py",
    "nlpgnn_full_proj/tests/GNN/gnn_for_nlp/text_gcn.py",
    "nlpgnn_full_proj/tests/GNN/gnn_for_nlp/text_sage.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gan.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gcn.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gin.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_graphsage.py",
    "nlpgnn_full_proj/tests/KG2E/run_tucker.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/albert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/albert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/bert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/bert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/data_processing.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_crf_test.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_crf_train.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/ner_data_preprocess.py",
    "nlpgnn_full_proj/tests/TG/EN/generation.py",
    "nlpgnn_full_proj/tests/TG/EN/interactive.py"
  };

  /**
   * The complete MusicTransformer-tensorflow2.0 subject vendored verbatim under {@code
   * musictransformer_proj} (wala/ML#683, wala/ML#684), so its whole-project guards analyze the
   * subject shape rather than a distilled fixture.
   */
  private static final String[] MUSICTRANSFORMER_PROJECT_FILES = {
    "musictransformer_proj/custom/callback.py",
    "musictransformer_proj/custom/layers.py",
    "musictransformer_proj/data.py",
    "musictransformer_proj/deprecated/seq_test.py",
    "musictransformer_proj/deprecated/sequence.py",
    "musictransformer_proj/deprecated/train.py",
    "musictransformer_proj/dist_train.py",
    "musictransformer_proj/generate.py",
    "musictransformer_proj/model.py",
    "musictransformer_proj/params.py",
    "musictransformer_proj/preprocess.py",
    "musictransformer_proj/train.py",
    "musictransformer_proj/utils.py"
  };

  /**
   * Verbatim whole-project guard for <a
   * href="https://github.com/wala/ML/issues/690">wala/ML#690</a>: the full NLPGNN subject vendored
   * as-is (all 94 {@code .py} files, matching the consumer's whole-project run; no added {@code
   * __init__.py}s under {@code tests/}). Before the fix, whichever same-named {@code GenGPT2}
   * sibling's closure reached the shared {@code sample_sequence.step} node second lost its lexical
   * {@code model} wiring (WALA's one-shot {@code visitLexical} snapshot), so its {@code
   * predict}/{@code call} never dispatched and its method nodes vanished at whole-project scale.
   * This pins the generation sibling's {@code predict} node and its parameter type, which must be
   * symmetric with {@link #testNlpgnnFullInteractive()}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    runNlpgnnFullGeneration();
  }

  /**
   * Runs the NLPGNN whole-project generation analysis with its call-graph and type assertions.
   * Package-visible so {@link DiagnosticLoggingVolumeTest} can rerun the analysis under {@code
   * FINEST} without invoking another class's {@code @Test} method.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  public void runNlpgnnFullGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "tests/TG/EN/generation.py",
        "GenGPT2.predict",
        "nlpgnn_full_proj",
        1,
        1,
        Map.of(
            3, Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * In-vivo anchor for wala/ML#704: the vendored NLPGNN {@code einsum_via_matmul} ({@code
   * nlpgnn/layers/dense.py}). The {@code input_tensor} parameter now carries concrete batch and
   * sequence dimensions with a dynamic trailing (hidden) dimension, delivered from the entry
   * scripts' explicit {@code model.build} contracts through the embedding's output reshape
   * (wala/ML#716, wala/ML#717): {@code (8, 100)} and {@code (8, 10)} leading pairs from the entries
   * whose pipelines reach this layer, each with the trailing {@code input_shape[-1] *
   * self.embedding_size} element unresolved, since the factor comes from a checkpoint config the
   * analysis cannot read — a fixed runtime size of unknown value ({@link UnresolvedDim},
   * wala/ML#721), not a runtime-{@code None} axis. The rank-2 {@code (8, D)} member is the
   * embedding guard-φ's path-insensitive phantom (the pre-{@code expand_dims} member). The {@code
   * tf.reshape}/{@code tf.squeeze} producer registrations and the callee-return descent for
   * layer-call results add the degraded-rank members ({@code (D, D)}, {@code (D, D, D)}, {@code (8,
   * D, D)}): the einsum body's own reshapes now compute generator-side through the {@code
   * get_shape_list} walk, whose non-entry contexts resolve rank but not every dimension. The rank-4
   * {@code (8, 100, U, U)}/{@code (8, 10, U, U)} members are the {@code DenseLayer3dProj} contexts'
   * inputs (the attention's return value): the worklist engine converges the loop-carried union
   * from its non-cyclic base and all four proj contexts carry them (wala/ML#365 Phase 3 resolved
   * the fourth, the wala/ML#718 residual under the retired round-based resolution). The formerly
   * shape-⊤ members carry equation-proven ranks since the einsum-operand refinement (wala/ML#704):
   * {@code DenseLayer3d.call}'s {@code use_einsum} arm makes its input an operand of the rank-3
   * {@code "BFH"} term and {@code DenseLayer3dProj.call}'s of the rank-4 {@code "BFND"} term, and
   * the refined parameter states transport through the call boundary into this helper; the
   * dead-site rank-2/3 matmul artifacts are gone with the caller-walk filtering (wala/ML#763).
   * Every proven axis stays {@link UnresolvedDim} in vivo, since {@code w}'s extents are
   * config-derived; the union is dtype-homogeneous {@code float32} since the dtype feed
   * (wala/ML#736) replaced the attention path's pure-⊤ seeds. The {@code w} parameter keeps rank 3
   * and {@code float32} (its chain is layer-local) but no numeric dimensions, since the {@code
   * build}-computed head sizes also derive from the config.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullEinsumViaMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/dense.py",
        "einsum_via_matmul",
        "nlpgnn_full_proj",
        2,
        14,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(new NumericDim(8), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(10), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(100), UnresolvedDim.INSTANCE)),
                // The wala/ML#737 partial composition proves the rank-4 attention form's batch
                // axis even when the remaining operand axes stay unresolved.
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(10),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(100),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE))),
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))))));
  }

  /**
   * In-vivo anchor for wala/ML#736 at its measurement point: the vendored NLPGNN {@code
   * DenseLayer3d.call}'s {@code input_tensor}, formerly carrying an {@code unknown}-dtype member
   * alongside its {@code float32} ones (the heterogeneous union the issue reported). The dtype feed
   * replaces pure-⊤ generator seeds with synthetic edges from their dtype-source operands, so the
   * attention path's element-wise and pass-through results take the converged {@code float32}
   * instead of seeding {@code unknown}, and the union is dtype-homogeneous in every context. The
   * {@code (8, 100)}/{@code (8, 10)} leading pairs are the entry contracts (wala/ML#717); the ranks
   * are the einsum operand refinement's (wala/ML#704).
   *
   * <p>The former rank-4 {@code (8, 100, ?, ?)}/{@code (8, 10, ?, ?)} members and the rank-2/3
   * dead-arm artifacts are gone (wala/ML#763): with the {@code use_einsum} guard folding
   * (wala/ML#761, wala/ML#762), the dead {@code einsum_via_matmul} sites no longer contribute, to
   * either the caller walks or the dataflow φs. The {@code (8, ?, ?, ?)} member is the legitimate
   * wala/ML#737 partial.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullDense3dInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/dense.py",
        "DenseLayer3d.call",
        "nlpgnn_full_proj",
        1,
        9,
        Map.of(
            3,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(8), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(100), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(10), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                // The wala/ML#737 partial composition proves the rank-4 attention form's batch
                // axis even when the remaining operand axes stay unresolved.
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)))));
  }

  /**
   * In-vivo anchor for the wala/ML#766 GNN entry feed: the vendored NLPGNN {@code GCNLayer.call}'s
   * {@code node_embeddings}, whose only runtime feed is the {@code Planetoid} loader's
   * row-normalized features. With {@code norm=True} constant at every {@code Planetoid} site, the
   * wala/ML#763 φ-arm suppression correctly cuts the un-normalized {@code np.array} arm, so the
   * parameter's type must arrive through the normalization itself: {@code
   * sp.diags(r_inv).dot(features)}, typed by the SciPy sparse product modeling ({@code scipy.xml}
   * plus {@code SparseMatrixDot}) as {@code ? of float32}: dtype from the dense operand, shape
   * unknown because the dense operand's extents are pickle-loaded data.
   *
   * <p>TODO: The shape should improve to {@code (Unresolved, Unresolved)} once rank flows through
   * the dense-ification chain ({@code todense}); see <a
   * href="https://github.com/wala/ML/issues/768">wala/ML#768</a>.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullGcnCallInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/models/GCN.py",
        "GCNLayer.call",
        "nlpgnn_full_proj",
        1,
        5,
        Map.of(3, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * In-vivo anchor for the wala/ML#704 einsum-operand refinement at its reopen measurement point:
   * the vendored NLPGNN {@code DenseLayer3dProj.call}'s {@code input_tensor}, formerly {@code ? of
   * float32} (shape ⊤) in every context. The {@code use_einsum} arm's {@code
   * einsum("BFND,NDH->BFH", input_tensor, w)} proves the input rank 4; the shared {@code N}/{@code
   * D} extents stay {@link UnresolvedDim} in vivo because {@code w}'s dimensions are config-derived
   * (contrast {@link #testDense3dProj()}, where the weight's static extents transfer). The {@code
   * (8, 100, U, U)}/{@code (8, 10, U, U)} members arrive already rank-4 from the entry contracts
   * (wala/ML#717) and pass through the refinement with their concrete leading dims intact.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullDenseProjInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/dense.py",
        "DenseLayer3dProj.call",
        "nlpgnn_full_proj",
        1,
        8,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(100),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(10),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)))));
  }

  /**
   * In-vivo anchor for the wala/ML#716 exactness mode and the wala/ML#717 contract seed: the
   * vendored NLPGNN {@code WDEmbedding.call}'s {@code input_ids} parameter resolves to each entry
   * script's declared {@code model.build(input_shape=(3, batch_size, maxlen))} contract, delivered
   * through the {@code tf.split}/{@code tf.squeeze}/{@code tf.cast} chain of the wrapper's {@code
   * call}. The expected set is the union across the vendored entry scripts' contracts (the helper
   * unions per value number across calling contexts), each contract's leading stack dimension
   * divided out by the split and squeezed away, leaving the per-entry {@code (batch_size, maxlen)}
   * pairs; all are {@code int32} after the cast.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullEmbeddingInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/embedding.py",
        "WDEmbedding.call",
        "nlpgnn_full_proj",
        1,
        12,
        Map.of(
            3,
            Set.of(
                TensorType.of(INT_32, 8, 100),
                TensorType.of(INT_32, 16, 100),
                TensorType.of(INT_32, 8, 10),
                TensorType.of(INT_32, 1, 512),
                TensorType.of(INT_32, 6, 128),
                TensorType.of(INT_32, 2, 4),
                TensorType.of(INT_32, 2, 10))));
  }

  /**
   * Sibling half of {@link #testNlpgnnFullGeneration()} (wala/ML#690) — the {@code interactive}
   * entry script, the one whose {@code predict}/{@code call} nodes vanished in the consumer's
   * whole-project run. Its {@code predict} parameter typing must be symmetric with the generation
   * sibling's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullInteractive()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "tests/TG/EN/interactive.py",
        "GenGPT2.predict",
        "nlpgnn_full_proj",
        1,
        1,
        Map.of(
            3, Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/676">wala/ML#676</a> subject: {@code
   * DynamicPositionEmbedding.call}'s {@code inputs} parameter on the vendored {@code
   * jason9693/MusicTransformer-tensorflow2.0} {@code custom/layers.py}. With {@code
   * tf.keras.layers.Embedding} modeled and constructor keyword arguments forwarded (wala/ML#664),
   * the chain composes concretely: tokens {@code (2, 50)} int32 through the embedding to {@code (2,
   * 50, 64)} float32 into the position encoding. On 0.52.13 the parameter was tensor- classified
   * but shapeless ({@code ? of unknown}, an integer-arithmetic seed leaking through the pointer
   * analysis); wala/ML#664 removed the leak and this modeling supplies the honest chain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   *     <p>The local count excludes the two all-constant slice-constructor objects under the
   *     multi-dim subscript, pinned non-tensor since wala/ML#732; the subscript result stays typed.
   */
  @Test
  public void testMusicTransformerPositionEmbedding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "musictx_proj/custom/__init__.py",
          "musictx_proj/custom/layers.py",
          "musictx_proj/tf2_test_musictx_encoder.py"
        },
        "custom/layers.py",
        "DynamicPositionEmbedding.call",
        "musictx_proj",
        1,
        4,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 50, 64))));
  }

  /**
   * Pins {@code TextCNN.call(inputs, training)}'s tensor-parameter type. The {@code TextCNN} model
   * is vendored verbatim from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/TextCNN.py}); only
   * the driver is bespoke. This is a real-world text-classification utility (a convolutional
   * sentence encoder: embedding lookup, parallel {@code Conv1D} kernels, global average pooling,
   * concatenation, batch normalization, and a softmax dense head), exercised for tensor-type
   * inference coverage. Unlike the GNN cohort ({@link #testGcnCall()}, {@link #testGatCall()}) and
   * the float-feature method archetypes, the decorated parameter {@code inputs} is an integer
   * token-ID tensor (an embedding-lookup index), so this measures int-dtype parameter recovery.
   *
   * <p>The tensor parameter {@code inputs} is recovered concretely on both axes &mdash; {@code (2,
   * 5) int32} &mdash; flowing from the driver's {@code model(inputs, training=False)} call site
   * through {@code tf.keras.Model.__call__} dispatch into {@code TextCNN.call}. As with the GNN
   * cohort, the decorated function's input signature &mdash; the analysis goal &mdash; is exact;
   * here it confirms that recovery holds for an integer-dtype parameter and a convolutional
   * (non-message-passing) body, not only the float-feature archetypes.
   *
   * <p>The tracked tensor variables are {@code inputs} (concrete {@code (2, 5) int32}), the
   * embedding output (concrete {@code (2, 5, 8)} float32 with {@code Embedding} modeled,
   * wala/ML#676), and the softmax {@code Dense} head's output ({@code float32} dtype but ⊤ shape
   * &mdash; {@code DenseCall} hard-codes {@code float32} but loses the shape through the
   * chained-layer body, <a href="https://github.com/wala/ML/issues/358">wala/ML#358</a>/<a
   * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>). The convolution intermediate is
   * ⊥: {@code Conv1D}/{@code GlobalAvgPool1D} remain unmodeled, and its former both-axes-⊤ rode the
   * spurious unknown-tensor composition of the enumerate over the unresolved {@code kernel_sizes}
   * list, which wala/ML#730 removed. These residual body locals are pre-existing modeling gaps, not
   * new findings, and are downstream of the (exact) input signature.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTextcnnCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "textcnn_proj/nlpgnn/__init__.py",
          "textcnn_proj/nlpgnn/models/__init__.py",
          "textcnn_proj/nlpgnn/models/TextCNN.py",
          "textcnn_proj/tf2_test_textcnn_call.py"
        },
        "nlpgnn/models/TextCNN.py",
        "TextCNN.call",
        "textcnn_proj",
        1,
        3,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>'s gpt-2
   * case: a callee parameter that receives a tensor argument at a direct method-call site is typed
   * by Ariadne. The {@code Gpt2} model, dataset pipeline, and {@code _train_step}/{@code
   * train_step}/{@code get_loss} dispatch are vendored verbatim from {@code
   * akanyaani/gpt-2-tensorflow2.0} ({@code gpt2_model.py}, {@code data_pipeline.py}); only the
   * transformer {@code layers} (and the {@code utils}/{@code scripts} helpers) are stubbed to
   * pass-throughs, since {@code get_loss}'s parameter typing does not depend on the model body.
   *
   * <p>A {@code padded_batch} dataset element flows through {@code fit} to the {@code
   * @tf.function(input_signature=...)}-decorated {@code train_step}, then {@code _train_step}, then
   * {@code get_loss(targets, predictions)}. The {@code real} parameter (vn=3), bound to the
   * dataset-sourced {@code targets}, types to {@code (2, 2)} int32, so Ariadne emits the parameter
   * type for this exact shape. With wala/ML#665 forwarding wildcard import bindings, {@code
   * pred} types too: the stubbed model body's forward output is a rank-3 union with the vocab
   * dimension recovered. This pins that wala/ML#618's residual gpt-2 failure is downstream of
   * Ariadne, not an emission gap.
   */
  @Test
  public void testGpt2InterprocGetLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_proj/layers/__init__.py", "gpt2_proj/layers/embedding_layer.py",
          "gpt2_proj/layers/feed_forward.py", "gpt2_proj/layers/layer_norm.py",
          "gpt2_proj/layers/attention_layer.py", "gpt2_proj/utils/__init__.py",
          "gpt2_proj/utils/tf_utils.py", "gpt2_proj/scripts/__init__.py",
          "gpt2_proj/scripts/utils.py", "gpt2_proj/data_pipeline.py",
          "gpt2_proj/gpt2_model.py", "gpt2_proj/tf2_test_gpt2_probe.py"
        },
        "gpt2_model.py",
        "Gpt2.get_loss",
        "gpt2_proj",
        2,
        8,
        Map.of(
            3,
            Set.of(TensorType.of(INT_32, 2, 2)),
            4,
            // The model forward output: rank 3 with the vocab dimension recovered as the
            // constant 100; the dtype is refinable once `add_weight` consumes its `dtype`
            // argument. The (2, 2) int32 member is the call-site union with the label tensor.
            Set.of(
                new TensorType(
                    UNKNOWN,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(100))),
                TensorType.of(INT_32, 2, 2))));
  }

  /**
   * The full subject from {@code akanyaani/gpt-2-tensorflow2.0} (a perf-eval corpus subject)
   * vendored verbatim with the REAL transformer layers and {@code input_fn}, unlike the stubbed
   * {@link #testGpt2InterprocGetLoss()}. {@code get_loss}'s {@code real} now types end to end
   * (wala/ML#618), resolving the whole dataset-sourced chain that previously dehybridized it.
   *
   * <p>The chain: {@code real} (the dataset-sourced {@code targets}) is built in {@code input_fn}
   * as {@code
   * tf.data.TFRecordDataset(...).map(parse_example).padded_batch(...).repeat(...).prefetch(...)},
   * passed as a list {@code _model.fit([_train, _test], ...)}, list-unpacked, iterated with {@code
   * enumerate} and nested unpacking, and forwarded through an indirected {@code train_fuc} into
   * {@code train_step} → {@code _train_step} → {@code get_loss}. Each layer is modeled: {@code
   * parse_example} densifies a {@code tf.io.VarLenFeature} through a dict to {@code (?,)} int32
   * (wala/ML#646, pinned by {@link #testParseExampleTuple()}); {@code map} types the element from
   * {@code map_func}'s return (wala/ML#506, {@link #testDatasetMapTuple()}); a pass-through
   * transform after {@code map} keeps the mapped type (wala/ML#649, {@link
   * #testDatasetMapRepeat()}); {@code TFRecordDataset} is chainable ({@link #testTfrecordMap()});
   * the dataset survives the list (wala/ML#648, {@link #testFitLoop()}); and the {@code
   * padded_batch} dims apply (wala/ML#673) through the tuple element and the post-batch
   * pass-throughs (wala/ML#759, {@link TestDatasets#testPaddedBatchTupleEnumerate()}). So {@code
   * real} resolves to the batched element {@code (32, Dynamic)} int32 (the pipeline's {@code
   * batch_size=32} default with the pad-to-longest sequence axis), unioned with the standard
   * partial-batch sibling {@code (?, Dynamic)}.
   *
   * <p>{@code pred} types too (wala/ML#665): the model forward output is a tensor union. With
   * {@code add_weight} consuming its {@code shape}/{@code dtype} arguments (wala/ML#667),
   * constructor keyword arguments forwarded to {@code __init__} (wala/ML#664), the wala/ML#739
   * operand-walk repairs, and parameter defaults materializing in the pointer analysis
   * (wala/ML#743), the decoder stack resolves end to end under the fit-path contexts: the logits
   * forms are {@code (32, Dynamic, 10)} float32 and its partial-batch sibling {@code (?, Dynamic,
   * 10)} (the batch axis rides the wala/ML#759-batched dataset element, the sequence axis the
   * dataset's dynamic pad), alongside the {@code (?, ?, 10)} partial. The wala/ML#680 {@code
   * unknown}-dtype phantom is gone: with the decoder-stack output resolving, {@code
   * OutputLayer.call}'s dead {@code self.porj_weights} arm no longer contributes a member. The
   * union is the order-independent fixed point (wala/ML#674): identical across runs and across
   * suite/single-test modes. Analyzed statically here, like the consumer's vendoring; it runs in
   * the perf-eval with its tfrecord/data setup.
   *
   * <p>The former {@code (32, Dynamic, 8, 8)}/{@code (?, Dynamic, 8, 8)} members, the {@code
   * mode="projection"} call's rank-3 input crossing into the embedding-mode lookup, are gone: <a
   * href="https://github.com/wala/ML/issues/746">wala/ML#746</a>'s per-call-site arm filtering
   * prunes the embedding arm at that site.
   */
  @Test
  public void testGpt2GetLossVendored()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py"
        },
        "A.py",
        "Gpt2.get_loss",
        "gpt2_vendored",
        2,
        8,
        Map.of(
            3,
            Set.of(
                new TensorType(INT_32, asList(new NumericDim(32), DynamicDim.INSTANCE)),
                new TensorType(INT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE))),
            4,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(10))),
                new TensorType(
                    FLOAT_32, asList(new NumericDim(32), DynamicDim.INSTANCE, new NumericDim(10))),
                new TensorType(
                    FLOAT_32,
                    asList(new SymbolicDim("?"), DynamicDim.INSTANCE, new NumericDim(10))))));
  }

  /**
   * Companion to {@link #testGpt2InterprocGetLoss()} that drives the <em>distributed</em> reach of
   * <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>'s gpt-2 case: the same vendored
   * {@code Gpt2} model, but reached via {@code distributed_train_step} &rarr; {@code
   * _distributed_train_step} &rarr; {@code mirrored_strategy.run(step_fn, args=(inputs, targets))}
   * &rarr; {@code step_fn} &rarr; {@code get_loss(tar, logits)}, the path the real subject takes.
   *
   * <p>{@code get_loss}'s {@code real} parameter (vn=3), bound to the dataset-sourced {@code
   * targets} that flows through {@code strategy.run}'s {@code args} tuple into {@code step_fn}'s
   * {@code tar}, types to {@code (2, 2)} int32 exactly as in the direct reach. This exercises both
   * halves of the wala/ML#618 distributed-reach fix: the {@code tensorflow/distribute/run/run}
   * model forwarding both tuple elements (see {@link #testStrategyRunTwoArgsInp()}), and the {@code
   * args} parameter name surviving summary loading (the <a
   * href="https://github.com/wala/WALA/pull/1972">wala/WALA#1972</a> fix to {@code
   * XMLMethodSummaryReader}'s name filter), without which the keyword {@code args=} could not bind.
   */
  @Test
  public void testGpt2DistributedGetLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_proj/layers/__init__.py", "gpt2_proj/layers/embedding_layer.py",
          "gpt2_proj/layers/feed_forward.py", "gpt2_proj/layers/layer_norm.py",
          "gpt2_proj/layers/attention_layer.py", "gpt2_proj/utils/__init__.py",
          "gpt2_proj/utils/tf_utils.py", "gpt2_proj/scripts/__init__.py",
          "gpt2_proj/scripts/utils.py", "gpt2_proj/data_pipeline.py",
          "gpt2_proj/gpt2_model.py", "gpt2_proj/tf2_test_gpt2_distributed_probe.py"
        },
        "gpt2_model.py",
        "Gpt2.get_loss",
        "gpt2_proj",
        2,
        8,
        Map.of(
            3,
            Set.of(TensorType.of(INT_32, 2, 2)),
            4,
            // The model forward output: rank 3 with the vocab dimension recovered as the
            // constant 100; the dtype is refinable once `add_weight` consumes its `dtype`
            // argument. The (2, 2) int32 member is the call-site union with the label tensor.
            Set.of(
                new TensorType(
                    UNKNOWN,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(100))),
                TensorType.of(INT_32, 2, 2))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at subject scale: in the
   * vendored MusicTransformer-tensorflow2.0 whole-project analysis, {@code
   * MusicTransformer.__prepare_train_data}'s direct {@code tf.ones((y.shape[0], 1), dtype=y.dtype)}
   * must be visible through the {@code tf} binding carried by {@code from custom.layers import *}
   * (a wildcard re-export of an import binding). What is asserted is the function's tensor-variable
   * census: seven distinct value numbers, which include the {@code tf.ones} result and its
   * downstream locals (observed as the runtime-true {@code (Dynamic, 1)} of float32 at vn 6);
   * pre-fix, only the two ⊤-typed parameters survived, so any regression of the binding collapses
   * the count. The vendored file is verbatim, so no {@code consume} sink can pin the local's exact
   * type here; the exact-type pins live in the fixture-scale probes ({@link #testWildcardUsedTf()},
   * {@link #testWildcardPkgNoInitTf()}). (The sibling {@code
   * MusicTransformerDecoder.__prepare_train_data} has no live {@code tf} use; all its tensor lines
   * are commented out in the subject.) Before the wala/ML#683 {@code tensorflow.python} namespace
   * binding, the empty {@code keras} binding that {@code custom/layers.py} re-exports starved the
   * whole chain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   *     <p>The local count excludes the two all-constant slice-constructor objects under the {@code
   *     [:, :-1]} subscripts, pinned non-tensor since wala/ML#732.
   */
  @Test
  public void testMusicTransformerPrepareTrainData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        MUSICTRANSFORMER_PROJECT_FILES,
        "model.py",
        "MusicTransformer.__prepare_train_data",
        "musictransformer_proj",
        2,
        5,
        Map.of(
            2,
            Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE),
            3,
            Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }
}
