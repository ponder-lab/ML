package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a key read on a component of a zipped dataset's element types the zipped dataset's
 * component: the component is selected by index from a tuple of datasets, or by key from a dict of
 * datasets, and the key then selects within that dataset's dict element. Before, the read took the
 * union of the selected dataset's components with a dimension peeled off each, which excluded the
 * component's type.
 */
public class TestDatasetZipDict extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_zip_dict.py";

  /** A dict dataset zipped with another dataset, selected by index and then by key. */
  @Test
  public void testTupleOfDatasets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "tuple_sink", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** Two dict datasets zipped, the second's key read. */
  @Test
  public void testPairOfDictDatasets()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "pair_sink", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2))));
  }

  /** A dict of datasets, a dataset selected by key and then a key of its element. */
  @Test
  public void testDictOfDatasets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "named_sink", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** A dict of datasets, a dataset of scalars selected by key. */
  @Test
  public void testDictOfDatasetsScalar()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "named_label_sink", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64))));
  }

  /** The zipped dict datasets batched, the first's key read. */
  @Test
  public void testBatchedTupleOfDatasets()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "batched_sink",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3),
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), new NumericDim(3))))));
  }
}
