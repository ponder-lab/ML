package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a generator argument whose shape is unknown degrades the result instead of crashing
 * the analysis (wala/ML#978). {@code getShapesOfValue} returns {@code null} (⊤) for such a value,
 * and several generators dereferenced it. The argument here is a {@code tf.where} position vector,
 * whose length depends on the data. Before the fix each fixture's analysis threw a {@code
 * NullPointerException}, so every one of these tests failed.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestUnknownShapeArgument extends AbstractTensorTest {

  private static final String RAGGED = "tf2_test_unknown_shape_argument.py";

  private static final String OTHER = "tf2_test_unknown_shape_argument2.py";

  /**
   * {@code tf.RaggedTensor.from_row_starts} with row starts of unknown shape, the issue's case: the
   * row count reads as dynamic, as it does when the argument gives no evidence at all.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRowStarts()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(RAGGED, "consume_row_starts", 1, 1, Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * {@code tf.RaggedTensor.from_row_limits} with row limits of unknown shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRowLimits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(RAGGED, "consume_row_limits", 1, 1, Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * {@code tf.RaggedTensor.from_row_splits} with row splits of unknown shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRowSplits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(RAGGED, "consume_row_splits", 1, 1, Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * {@code tf.RaggedTensor.from_row_lengths} with row lengths of unknown shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(RAGGED, "consume_row_lengths", 1, 1, Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * A Keras {@code Flatten} over an input of unknown shape reads as an unknown shape, in the
   * input's float32. The layer call is {@code FlattenCall}'s, which already declined on an unknown
   * input shape; this pins that it keeps doing so. The TF1 {@code tf.layers.flatten} generator,
   * {@code Flatten}, carries the same guard, but no summary allocates its type, so no fixture can
   * reach it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFlatten()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(OTHER, "consume_flatten", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * {@code tf.random.poisson} with a rate of unknown shape: the rate's shape is the output's
   * trailing axes, so the output shape is unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPoisson()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(OTHER, "consume_poisson", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The element of {@code tf.data.Dataset.from_tensors} over a tensor of unknown shape reads as a
   * tensor of unknown type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDatasetFromTensors()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        OTHER,
        "consume_dataset_element",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * The element of {@code tf.data.Dataset.choose_from_datasets} over a dataset whose element has an
   * unknown shape reads as a tensor of unknown type: an unknown input shape is ⊤ for the union of
   * the input datasets' element shapes, not an absent member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDatasetChooseFromDatasets()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        OTHER,
        "consume_chosen_element",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * The element of {@code tf.data.Dataset.sample_from_datasets} over a dataset whose element has an
   * unknown shape reads as a tensor of unknown type: an unknown input shape is ⊤ for the union of
   * the input datasets' element shapes, not an absent member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDatasetSampleFromDatasets()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        OTHER,
        "consume_sampled_element",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * {@code tf.RaggedTensor.from_row_starts} whose values, not its partition, have an unknown shape:
   * the result's trailing axes, and so its rank, are unknown, so its shape is ⊤. Every
   * row-partition constructor builds its shape through {@code
   * RaggedTensorFromValues.constructRaggedShape}, which dereferenced the unknown values shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnknownValuesRowStarts()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        RAGGED,
        "consume_unknown_values",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * {@code tf.RaggedTensor.from_value_rowids} with values of unknown shape, through the same shape
   * construction as {@link #testUnknownValuesRowStarts}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnknownValuesValueRowIds()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        RAGGED,
        "consume_unknown_values_rowids",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }
}
