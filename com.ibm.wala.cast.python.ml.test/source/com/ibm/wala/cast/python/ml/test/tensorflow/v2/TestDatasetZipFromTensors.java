package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A zip of datasets built by {@code from_tensors} reads each member of their elements at its own
 * type, as the datasets themselves do (<a
 * href="https://github.com/wala/ML/issues/1211">wala/ML#1211</a>).
 */
public class TestDatasetZipFromTensors extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_zip_from_tensors.py";

  /**
   * A member of the pair the dataset itself yields.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnzippedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unzipped_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The first member of the zipped pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZippedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zipped_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The second member of the zipped pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZippedB() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zipped_b", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * The element of the {@code from_tensors} dataset zipped beside the pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZippedC() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zipped_c", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A flat dataset zipped with a {@code from_tensors} dataset keeps its own element's type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testZippedFlat() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zipped_flat", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 5))));
  }
}
