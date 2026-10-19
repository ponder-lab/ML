package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_7_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_64_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_64_7_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of the {@code Model.getWeightShapes} weight-graph walk ({@code testModelAttributes*}),
 * carved from the {@code TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestModelWeightGraph extends AbstractTensorTest {

  @Test
  public void testModelAttributes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes2.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes3.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes4.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes5.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes6.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Tests precise dataflow tracing for Keras {@code Model} weights when using keyword arguments for
   * both {@code Dense} layer instantiation and {@code Model} construction. Verifies that the
   * weights are correctly identified and have the expected shapes {@code (64, 5)} and {@code (5,)}.
   */
  @Test
  public void testModelAttributes7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes7.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Multi-Model precision regression: two distinct {@code tf.keras.models.Model(...)} calls in one
   * fixture, two sink functions, disjoint shapes per model. Validates that under the current
   * modeling each sink's parameter sees only its own model's weight shapes (not the union across
   * both models). See wala/ML#380's discussion of `Model.read_data` materialization. Companion to
   * {@link #testModelAttributesMultiModel2()} (same fixture, second sink). Disjoint dim choices
   * (64/5 vs 32/7) make a precision regression mechanically detectable: a "shapes unioned across
   * models" failure mode produces the 4-element set, not a 2-element subset.
   */
  @Test
  public void testModelAttributesMultiModel()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testModelAttributesMultiModel()} — pins the second sink's parameter to the
   * second model's weight shapes only.
   */
  @Test
  public void testModelAttributesMultiModel2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(TENSOR_32_7_FLOAT32, TENSOR_7_FLOAT32)));
  }

  /**
   * Multi-Model separation with one extra call-chain frame: both Models are constructed inside a
   * {@code make_model(units)} helper, so both user-side calls of {@code make_model(...)} share the
   * same call site for {@code Model.do} (the one inside {@code make_model}). Call strings alone
   * collapsed both user models into one allocation context, unioning both models' weight shapes at
   * every sink; the receiver-keyed trampoline contexts of <a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a> keep each model's dispatch chain
   * separate, so each sink now sees exactly its own model's weight shapes.
   */
  @Test
  public void testModelAttributesMultiModelWrapped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi_wrapped.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testModelAttributesMultiModelWrapped()} — same fixture, second sink,
   * pinned to the second model's weight shapes only (wala/ML#679).
   */
  @Test
  public void testModelAttributesMultiModelWrapped2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi_wrapped.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_7_FLOAT32, TENSOR_7_FLOAT32)));
  }

  /**
   * The weights of a Sequential-built model come from its layer list (<a
   * href="https://github.com/wala/ML/issues/832">wala/ML#832</a>'s weights half).
   *
   * <p>The same two-layer network as {@link #testModelAttributes2()}, with the same weights, built
   * from a layer LIST rather than from {@code inputs}/{@code outputs}. That test is the control:
   * the functional model's weights come from walking back from its {@code outputs}, which a
   * Sequential constructor does not take.
   *
   * <p>Every bias is {@code (units,)}, and every kernel after the first is {@code (units[i-1],
   * units[i])}, so three of the four shapes are determined by the list. The first kernel's input
   * width is the width the model is fed, a fixed size at run time that the list does not give (the
   * {@code Dense} summary has no {@code input_shape} parameter), so that axis is unresolved: {@code
   * (?, 4)}, where the program has {@code (3, 4)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSequentialWeightShapes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sequential_weights.py",
        "f",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, List.of(UnresolvedDim.INSTANCE, new NumericDim(4))),
                TENSOR_4_FLOAT32,
                TENSOR_4_5_FLOAT32,
                TENSOR_5_FLOAT32)));
  }

  /**
   * A Sequential model whose list holds a layer other than {@code Dense}, here a {@code Conv1D},
   * has weights its list does not give, so its weights are {@code float32} tensors of shapes the
   * analysis does not know, not the {@code Dense} layer's alone (<a
   * href="https://github.com/wala/ML/issues/832">wala/ML#832</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSequentialWeightShapesWithOtherLayer()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sequential_weights_mixed.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A Sequential model whose {@code Dense} layer's {@code units} may be {@code None}, as far as the
   * analysis knows, has weights of unknown shape: a {@code units} that does not resolve to sizes
   * alone gives no kernel or bias shape. Before, the {@code None} member crashed the analysis.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSequentialWeightShapesWithNoneUnits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sequential_weights_units_none.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The functional counterpart of {@link #testSequentialWeightShapesWithNoneUnits()}: a model whose
   * {@code Dense} layer's {@code units} may be {@code None} has weights of unknown shape. Before,
   * the {@code None} member crashed the analysis.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelWeightShapesWithNoneUnits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_weights_units_none.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
