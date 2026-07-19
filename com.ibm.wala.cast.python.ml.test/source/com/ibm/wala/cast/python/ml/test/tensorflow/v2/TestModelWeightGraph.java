package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_7_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_64_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_64_7_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_FLOAT32;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of the {@code Model.getWeightShapes} weight-graph walk ({@code testModelAttributes*}),
 * carved from the {@link TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
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
}
