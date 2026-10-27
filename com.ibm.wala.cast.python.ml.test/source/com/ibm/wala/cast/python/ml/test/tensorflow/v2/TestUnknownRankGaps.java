package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Values whose rank the analysis loses although it is fixed, each pinned at what the analysis reads
 * today, and the fixes that restore it. Each test names the issue whose fix moves it, and the type
 * it should then read.
 */
public class TestUnknownRankGaps extends AbstractTensorTest {

  /** A {@code float32} of a dynamic leading axis and 6, a segment reduction's output. */
  private static final TensorType DYNAMIC_6_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(6)));

  /** A {@code float32} tensor of unknown shape. */
  private static final TensorType UNKNOWN_FLOAT32 = new TensorType(FLOAT_32, null);

  /** A {@code float64} tensor of unknown shape. */
  private static final TensorType UNKNOWN_FLOAT64 = new TensorType(FLOAT_64, null);

  /**
   * An element of the result of calling a functional model whose outputs are a list.
   *
   * <p>TODO: wala/ML#1157: expect {@code (4, 13, 13, 3, 7)} of {@code float32}, the shape of {@code
   * outputs[1]}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testListOutputModelElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_list_output_model.py", "consume", 1, 1, Map.of(2, Set.of(UNKNOWN_FLOAT32)));
  }

  /**
   * An {@code np.zeros} whose shape tuple holds an element of an ndarray (wala/ML#1158): the shape
   * keeps its rank, and the two axes the ndarray element gives, 52 at run time, are unresolved.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testZerosNdarrayElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_ndarray_element.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(4),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        new NumericDim(3),
                        new NumericDim(7))))));
  }

  /**
   * An {@code np.zeros} whose shape position may be an ndarray element or a literal (wala/ML#1158):
   * the position keeps the literal beside the element's unresolved size.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testZerosNdarrayElementOrLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_ndarray_element.py",
        "consume_mixed",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 4, 26),
                new TensorType(FLOAT_32, asList(new NumericDim(4), UnresolvedDim.INSTANCE)))));
  }

  /**
   * The control for {@link #testZerosOfShapeElement()}: {@code np.zeros(5)} reads {@code (5,)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testZerosOfLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_unknown_int.py",
        "consume_literal",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 5))));
  }

  /**
   * An {@code np.zeros} whose shape is an ndarray's {@code .shape[0]}.
   *
   * <p>TODO: wala/ML#1159: expect rank 1 of {@code float64}, with an unresolved extent.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testZerosOfShapeElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_unknown_int.py", "consume_zeros", 1, 1, Map.of(2, Set.of(UNKNOWN_FLOAT64)));
  }

  /**
   * An {@code np.zeros} whose shape is {@code len(xs)}, an {@code int} of unknown value: rank 1 of
   * {@code float64}, with an unresolved extent (wala/ML#1159).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testZerosOfLength()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_unknown_int.py",
        "consume_length",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, asList(UnresolvedDim.INSTANCE)))));
  }

  /**
   * An unsorted segment sum whose {@code num_segments} is {@code tf.shape(x)[0]} over an {@code x}
   * of static shape {@code (5, 6)}, forwarded through a helper's parameter: TensorFlow's static
   * shape folds it to 5, as the element read does (wala/ML#1160).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSegmentSumOfRuntimeCount()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_segment_runtime_count.py",
        "consume_direct",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 5, 6))));
  }

  /**
   * An unsorted segment sum whose {@code num_segments} is {@code tf.shape(inputs)[0]} over a Keras
   * input of a dynamic batch axis: TensorFlow's static shape reports {@code None} there, so the
   * leading axis is dynamic (wala/ML#721, wala/ML#1160).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSegmentSumOfDynamicCount()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_segment_runtime_count.py",
        "consume_dynamic",
        1,
        1,
        Map.of(2, Set.of(DYNAMIC_6_FLOAT32)));
  }

  /**
   * An unsorted segment sum whose {@code num_segments} reaches the helper after a starred argument,
   * {@code aggregate_starred(*leading, tf.shape(x)[0], tf.shape(y)[0])}: the parameter's slot in
   * the call holds {@code tf.shape(y)[0]}, 7, while the parameter binds {@code tf.shape(x)[0]}, 5,
   * so the element isn't read through the call and the leading axis is unresolved (wala/ML#751,
   * wala/ML#1160).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSegmentSumAfterStarredArgument()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_segment_runtime_count.py",
        "consume_starred",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(6))))));
  }

  /**
   * An unsorted segment sum whose {@code num_segments} is {@code len(xs)}, a Python {@code int} of
   * unknown value: a fixed size the analysis did not compute, so the leading axis is unresolved
   * (wala/ML#721, wala/ML#1160).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSegmentSumOfLength()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_segment_runtime_count.py",
        "consume_counted",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(6))))));
  }

  /**
   * A softmax over a segment reduction chosen by a branch, whose {@code num_segments} is {@code
   * tf.shape(x)[0]}: rank 2, with 6 as its trailing axis (wala/ML#1160). At run time the leading
   * axis is 5, but these reductions are anchored on their summaries' own nodes rather than on a
   * source, so the element isn't read through the callers and the axis is unresolved.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testChosenSegmentReduction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_segment_runtime_count.py",
        "consume_chosen",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(6))))));
  }

  /**
   * The control for {@link #testEmbeddingOfUnreadDimension()}: an {@code Embedding} with a literal
   * {@code output_dim} keeps its rank.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEmbeddingOfLiteralDimension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_unread_dim.py",
        "consume_literal",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(128))))));
  }

  /**
   * An {@code Embedding} whose {@code output_dim} comes from the environment, through a layer
   * multiplying it by a constant: rank 3 of {@code float32}, two dynamic axes and an unresolved one
   * (wala/ML#1163).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEmbeddingOfUnreadDimension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_unread_dim.py",
        "consume_unread",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, UnresolvedDim.INSTANCE)))));
  }

  /**
   * An {@code Embedding} whose {@code output_dim} is an absl flag, through a layer multiplying it
   * by the flag. The flag's value has an empty points-to set, so the product's other operand is of
   * unknown kind, and the product is a {@code float32} tensor of unknown shape: the operand may be
   * an array that broadcasts to another. Before wala/ML#1163 gave the layer's input its rank, the
   * product read the same; reading it as no tensor would be a regression.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEmbeddingOfFlagDimension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_unread_dim.py",
        "consume_flagged",
        1,
        1,
        Map.of(2, Set.of(UNKNOWN_FLOAT32)));
  }

  /**
   * The layer's own input in both programs above: the {@code Embedding}'s output, rank 3 of {@code
   * float32}, two dynamic axes and an unresolved one (wala/ML#1163).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLayerInputOfUnreadDimension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_unread_dim.py",
        "MultiplyConstant.call",
        1,
        2,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, UnresolvedDim.INSTANCE)))));
  }
}
