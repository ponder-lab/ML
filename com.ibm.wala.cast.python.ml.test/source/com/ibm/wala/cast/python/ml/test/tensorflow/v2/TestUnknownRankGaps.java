package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Values whose rank the analysis loses although it is fixed, each pinned at what the analysis reads
 * today. Each test names the issue whose fix moves it, and the type it should then read.
 */
public class TestUnknownRankGaps extends AbstractTensorTest {

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
   * An {@code np.zeros} whose shape tuple holds an element of an ndarray.
   *
   * <p>TODO: wala/ML#1158: expect rank 5 of {@code float32}, {@code (4, 52, 52, 3, 7)} or with the
   * two axes the ndarray element gives unresolved.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testZerosNdarrayElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zeros_ndarray_element.py", "consume", 1, 1, Map.of(2, Set.of(UNKNOWN_FLOAT32)));
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
   * An {@code np.zeros} whose shape is {@code len(xs)}.
   *
   * <p>TODO: wala/ML#1159: expect rank 1 of {@code float64}, with an unresolved extent.
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
        Map.of(2, Set.of(UNKNOWN_FLOAT64)));
  }

  /**
   * An unsorted segment sum whose {@code num_segments} is {@code tf.shape(x)[0]}.
   *
   * <p>TODO: wala/ML#1160: expect rank 2 of {@code float32}, a dynamic leading axis and 6.
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
        Map.of(2, Set.of(UNKNOWN_FLOAT32)));
  }

  /**
   * A softmax over a segment reduction chosen by a branch, whose {@code num_segments} is {@code
   * tf.shape(x)[0]}.
   *
   * <p>TODO: wala/ML#1160: expect rank 2 of {@code float32}, a dynamic leading axis and 6.
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
        Map.of(2, Set.of(UNKNOWN_FLOAT32)));
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
   * multiplying it by a constant.
   *
   * <p>TODO: wala/ML#1163: expect rank 3 of {@code float32}, two dynamic axes and an unresolved
   * one.
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
        Map.of(2, Set.of(UNKNOWN_FLOAT32)));
  }
}
