package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Calls on {@code tf.keras.losses} instances (wala/ML#951): {@code
 * tf2_test_sparse_categorical_crossentropy.py} builds two {@code SparseCategoricalCrossentropy}
 * instances, one with {@code reduction="none"} and one with the default, and applies them to {@code
 * (2, 3, 10)} float32 logits against {@code (2, 3)} int32 labels; the fixture asserts each result's
 * shape and dtype at the sinks the tests pin.
 */
public class TestLosses extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_sparse_categorical_crossentropy.py";

  /** {@code reduction="none"}: the loss keeps the predictions' shape without the class axis. */
  @Test
  public void testSparseCategoricalCrossentropyNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_none", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The default reduction sums over the batch: a scalar of the predictions' dtype. */
  @Test
  public void testSparseCategoricalCrossentropyDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_scalar", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * The loss composes downstream: {@code tf.exp(tf.reduce_mean(loss_))}, the perplexity idiom, is a
   * float32 scalar. Before the loss call was modeled, every value computed from it read ⊤.
   */
  @Test
  public void testSparseCategoricalCrossentropyPerplexity()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_perplexity", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * A prediction typed only by the dataflow (a user layer's call result) types its loss through the
   * generator's {@code TRANSFORM} feed: {@code (2, 3, 4)} through a {@code (4, 10)} matmul gives
   * {@code (2, 3, 10)} logits, and the per-token loss is {@code (2, 3)} float32.
   */
  @Test
  public void testSparseCategoricalCrossentropyFed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_fed", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** {@code reduction="sum"} also gives a scalar: the non-{@code NONE} constant arm. */
  @Test
  public void testSparseCategoricalCrossentropySum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_sum", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * The decline: a {@code reduction} passed from a value the analysis cannot read (an environment
   * lookup) is neither the default nor {@code "none"}, so the loss keeps the predictions' dtype
   * with an unknown shape rather than asserting either of the two shapes it may have.
   */
  @Test
  public void testSparseCategoricalCrossentropyUnresolvedReduction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_unresolved", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Two instances with different reductions reaching one call: the call dispatches under a
   * receiver-keyed context per instance, each resolving its own {@code reduction}, so the union
   * across contexts is the two shapes, each true in its context, rather than ⊤. The generator's
   * disagreement arm is for a receiver whose contexts have collapsed, which this does not exercise.
   */
  @Test
  public void testSparseCategoricalCrossentropyTwoInstances()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_two_instances",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** One instance whose {@code reduction} is one of two literals: likewise ⊤. */
  @Test
  public void testSparseCategoricalCrossentropyTwoLiterals()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "consume_two_literals", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
