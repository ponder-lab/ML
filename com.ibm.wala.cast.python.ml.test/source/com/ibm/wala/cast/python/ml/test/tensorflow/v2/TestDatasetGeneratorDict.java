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
}
