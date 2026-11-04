package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A dataset element that is a namedtuple keeps its members as objects of their own, read by name as
 * by position (<a href="https://github.com/wala/ML/issues/1210">wala/ML#1210</a>).
 */
public class TestDatasetNamedTupleElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_namedtuple_element.py";

  /**
   * The element's first member read by its name.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNamedX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_named_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * The element's second member read by its name.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNamedM() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_named_m", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * The element's second member unpacked by its position.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnpackedM() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unpacked_m", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * The batched element's second member read by its name.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatchedM() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_batched_m", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * A member of an element built from a namedtuple constructed by keyword, read by its name.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeywordM() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword_m", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }
}
