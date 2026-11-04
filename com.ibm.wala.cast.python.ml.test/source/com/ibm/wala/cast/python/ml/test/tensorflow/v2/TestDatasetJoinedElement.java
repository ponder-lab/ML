package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A transformation whose receiver may be either of two datasets reads each member of its element
 * through the dataset the member came from, so a tensor from one keeps its own type and takes none
 * of the other's (<a href="https://github.com/wala/ML/issues/1199">wala/ML#1199</a>).
 */
public class TestDatasetJoinedElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_joined_element.py";

  /**
   * A batch of a shuffle: the flat dataset's tensor is batched, and the pair's first component is a
   * tuple, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_batched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6))));
  }

  /**
   * A shuffle alone, a pass-through.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testShuffled() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_shuffled", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6))));
  }

  /**
   * A batch of a shuffle over either of two flat datasets: both datasets' tensors.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFlats() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_flats",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), TensorType.of(FLOAT_32, 2, 5))));
  }

  /**
   * A batch of a shuffle over a flat dataset or one whose first component is a namedtuple: the flat
   * dataset's tensor alone, since a namedtuple is a structure, as a tuple is (<a
   * href="https://github.com/wala/ML/issues/1200">wala/ML#1200</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNamed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_named", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6))));
  }
}
