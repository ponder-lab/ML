package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests applying a {@code tf.keras.Model} subclass that builds a functional model in its
 * constructor and does not override {@code call}. Applying an instance runs {@code Model.call} over
 * the graph from the constructor's {@code inputs} to its {@code outputs}, so a {@code Dense(4)}
 * model applied to a {@code (2, 3)} input gives a {@code (2, 4)} output.
 */
public class TestFunctionalModelSubclass extends AbstractTensorTest {

  private static final String FILE = "tf2_test_functional_model_subclass.py";

  private static final TensorType TENSOR_2_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 4);

  /** The subclass passes {@code inputs} and {@code outputs} to its base by keyword. */
  @Test
  public void testKeyword() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The subclass passes {@code inputs} and {@code outputs} to its base by position. */
  @Test
  public void testPositional() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_positional", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The control: the same functional model constructed directly from {@code tf.keras.Model}. */
  @Test
  public void testDirect() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_direct", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** Calling {@code call} on a functional model directly runs the same graph. */
  @Test
  public void testDirectCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_call", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * A subclass that neither overrides {@code call} nor passes {@code outputs} to {@code
   * super().__init__} has no graph to run: applying an instance raises {@code NotImplementedError},
   * so the consumer receives no tensor.
   */
  @Test
  public void testNotFunctional() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_not_functional", 0, 0, Map.of());
  }

  /**
   * A subclass with no constructor of its own inherits {@code Model.__init__}, so the class's own
   * call passes {@code inputs} and {@code outputs}, here by keyword.
   */
  @Test
  public void testInheritedInitKeyword()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inherited_keyword", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * As {@link #testInheritedInitKeyword()}, with {@code inputs} and {@code outputs} by position.
   */
  @Test
  public void testInheritedInitPositional()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inherited_positional", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * A constructor that forwards {@code **kwargs} to {@code super().__init__}: the {@code outputs}
   * inside the dictionary are not read, so the result is a tensor of unknown shape, and not no
   * tensor, since the call does not demonstrably omit them.
   */
  @Test
  public void testKeywordsForwarded() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keywords_forwarded", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** As {@link #testKeywordsForwarded()}, for a constructor that forwards {@code *args}. */
  @Test
  public void testPositionalsForwarded()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_positionals_forwarded",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
