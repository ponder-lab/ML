package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the values of a training step reach a Keras layer's {@code call} typed (<a
 * href="https://github.com/wala/ML/issues/993">wala/ML#993</a>), hop by hop along the chain of
 * {@code tf2_test_encoder_training_path.py}: a {@code tf.function} step over a dataset of dicts
 * embeds {@code features["ids"]} through an inputter layer reached as a plain attribute, through a
 * {@code @property}, through the property's {@code getattr(..., default)} body, and through a
 * property inherited from a base class, then encodes the result through layers with and without
 * their own {@code __call__}. The embedding's shape is lost with the ids (see {@link
 * #testFeaturesIds()}), so the typed hops read as a {@code float32} tensor of unknown shape.
 */
public class TestEncoderTrainingPath extends AbstractTensorTest {

  private static final String FILE = "tf2_test_encoder_training_path.py";

  private static final TensorType IDS = TensorType.of(INT_32, 2, 3);

  /**
   * The dict element of a dataset of tuples of dicts is not typed, through the loop over the
   * dataset or through the {@code element_spec} signature, so {@code features["ids"]} reads as no
   * tensor. This is wala/ML#993's remainder on this chain.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#993 is fully fixed.
   */
  @Test(expected = AssertionError.class)
  public void testFeaturesIds() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_features_ids", 1, 1, Map.of(2, Set.of(IDS)));
  }

  /** The embedding lookup over the layer's weight types its result from the weight. */
  @Test
  public void testEmbedded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_embedded", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** The inputter reached as a plain attribute returns the embedding to the step. */
  @Test
  public void testStepInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_step_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The inputter reached through a {@code @property}: the instance's attribute holds the getter's
   * value, so calling it calls the layer (wala/ML#993). Before, the property's value was empty.
   */
  @Test
  public void testPropertyInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_property_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The inputter reached through a property whose body is {@code getattr(obj, "name", default)}:
   * the constant-name {@code getattr} reads as the attribute read, merged with its default.
   */
  @Test
  public void testGetattrInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_getattr_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** The inputter reached through a property declared on a base class. */
  @Test
  public void testInheritedPropertyInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inherited_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The inputter reached through a property that also has a setter of the same name, assigned
   * through that setter: the attribute holds the getter's value, and calling it calls the layer.
   */
  @Test
  public void testSettablePropertyInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_settable_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A property whose getter holds a plain function, called through the attribute with the
   * embedding: the function is called, so its result is the embedding.
   */
  @Test
  public void testSettableFunctionCall()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_settable_fn_out", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The setter of that property is reached only by the assignment in {@code __init__}, whose value
   * is a function, so its sink sees no tensor: a call through the attribute does not dispatch the
   * setter beside the getter's value.
   */
  @Test
  public void testSetterNotDispatchedByCall()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_setter_value", 0, 0);
  }

  /**
   * The encoder's own {@code __call__}, declared on its base class, is not dispatched: the call
   * goes straight to {@code call} (wala/ML#994). No type is lost on this chain by it.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#994 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDunderCallInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dunder_call_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** Control for wala/ML#994: the same {@code __call__} declared on the class itself dispatches. */
  @Test
  public void testDirectDunderCallInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_direct_dunder_call_inputs",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The encoder's {@code call} receives the embedding, with the extra positional {@code
   * sequence_length} and {@code training=True} beside it, at its fed dtype: {@code float32}, not
   * the {@code float64} that {@code inputs *= self.num_units**0.5} imposed before wala/ML#992.
   */
  @Test
  public void testCallInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_call_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** {@code build_mask} receives the scaled inputs from {@code call}. */
  @Test
  public void testMaskInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mask_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** Control: an encoder without its own {@code __call__} receives the same embedding. */
  @Test
  public void testPlainCallInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_plain_call_inputs", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
