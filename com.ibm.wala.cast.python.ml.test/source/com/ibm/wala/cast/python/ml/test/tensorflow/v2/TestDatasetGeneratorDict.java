package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a dataset built by {@code from_generator} with a dict of {@code output_types} types
 * each key of its element, kept through {@code padded_batch}, where a function reads the key and
 * expands it. Before, a string key into such an element resolved nothing, and the value read as a
 * tensor of unknown dtype.
 */
public class TestDatasetGeneratorDict extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_generator_dict.py";

  /**
   * A padded batch of 1-D {@code float32} values, expanded to rank 3: a full batch of two, or a
   * last, partial one.
   */
  private static final Set<TensorType> EXPANDED_BATCH =
      Set.of(
          new TensorType(
              FLOAT_32, asList(new NumericDim(2), DynamicDim.INSTANCE, new NumericDim(1))),
          new TensorType(
              FLOAT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE, new NumericDim(1))));

  /** A batch of rows of three {@code float32} values. */
  private static final TensorType BATCH_3 = TensorType.of(FLOAT_32, 2, 3);

  /** The dtypes alone, with the shapes from {@code padded_shapes}. */
  @Test
  public void testOutputTypesDict() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "types_only_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /** The dtypes with a dict of {@code output_shapes} too. */
  @Test
  public void testOutputTypesAndShapesDict()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "shaped_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /**
   * A dataset class's {@code from_generator}, {@code filter}, {@code padded_batch} and {@code
   * prefetch}, distributed and stepped by a trainer through a {@code tf.function} that runs the
   * step per replica.
   */
  @Test
  public void testTrainerStep() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_dataset_generator_trainer.py", "trainer_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /** A dataset an object holds in a field, enumerated in a method: the element is typed. */
  @Test
  public void testEnumerateFieldDataset()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_enumerate_field_dataset.py", "field_sink", 1, 2, Map.of(2, Set.of(BATCH_3)));
  }

  /** Control: the same dataset enumerated from a local variable. */
  @Test
  public void testEnumerateLocalDataset()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_enumerate_field_dataset.py", "local_sink", 1, 2, Map.of(2, Set.of(BATCH_3)));
  }

  /**
   * A key whose {@code padded_shapes} entry cannot be read: its shape is unknown, not the upstream
   * component's, which the padding may change.
   */
  @Test
  public void testUnreadPaddedShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "unread_padding_sink", 1, 2, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A trainer whose loader attribute is written three times, as a common base trainer writes it:
   * {@code None} at construction, the dataset a setter is passed, and the distributed dataset made
   * from the attribute's own value. The element of the enumerated loader is any of the datasets
   * written, so it is their join.
   */
  @Test
  public void testBaseTrainerLoader() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dataset_generator_trainer.py",
        "base_trainer_sink",
        1,
        2,
        Map.of(2, EXPANDED_BATCH));
  }

  /** The dtypes and shapes from a dict of specs, the {@code output_signature} form. */
  @Test
  public void testOutputSignatureDict()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "signature_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /**
   * An attribute written with a dataset and with a distributed view of another object's attribute:
   * the second write brings the other object's datasets, so it is no rewrite of this attribute's
   * own value. Its part resolves no element here, so the element is not typed, rather than typed as
   * the first write's dataset alone.
   */
  @Test
  public void testOtherObjectsAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_dataset_generator_trainer.py", "copier_sink", 0, 0, Map.of());
  }

  /** A padded batch without {@code padded_shapes}: each key pads to the longest in its batch. */
  @Test
  public void testPaddedBatchWithoutPaddedShapes()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "unpadded_sink",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)))));
  }

  /** A declared shape the reader cannot read: the key's shape is unknown, its dtype declared. */
  @Test
  public void testUnreadOutputShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "unread_shape_sink", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * An attribute two calls write datasets of dicts to, enumerated and read by key: the element is
   * either dataset's.
   */
  @Test
  public void testJoinedDatasetsByKey()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_join.py",
        "dict_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 2, 3))));
  }

  /**
   * An attribute two calls write datasets of tuples to, enumerated and read by index: the element
   * is either dataset's.
   */
  @Test
  public void testJoinedDatasetsByIndex()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_join.py",
        "tuple_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 2, 3))));
  }

  /**
   * The same joined tuple element expanded: the expansion reads the element's shapes and dtypes by
   * index from the join.
   */
  @Test
  public void testJoinedDatasetsExpanded()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_join.py",
        "expanded_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 1), TensorType.of(INT_32, 2, 3, 1))));
  }

  /** A dataset the class body assigns, enumerated through an instance and read by key. */
  @Test
  public void testClassAttributeLoader()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_class_attribute.py",
        "class_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * A dataset the class body assigns beside one an instance write stores: the element is either,
   * not the instance write's alone.
   */
  @Test
  public void testClassAndInstanceAttributeLoader()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_class_attribute.py",
        "mixed_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 2, 3))));
  }

  /**
   * A class of its own whose body assigns the same attribute name a {@code bool} dataset: its
   * element is that dataset's, and it reaches no other class's read.
   */
  @Test
  public void testUnrelatedClassAttributeLoader()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_class_attribute.py",
        "unrelated_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(BOOL, 2, 3))));
  }
}
