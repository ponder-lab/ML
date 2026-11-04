package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code map} calls its function with a tuple element's members as separate arguments and any other
 * element whole, and its element is what the function returns, so each member keeps its own type
 * (<a href="https://github.com/wala/ML/issues/164">wala/ML#164</a>).
 */
public class TestDatasetMapElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_map_element.py";

  /**
   * The function's first parameter is the tuple element's first member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testParamX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_param_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * Its second parameter is the second member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testParamIds() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_param_ids", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A function over a flat element gets the element whole.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testParamRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_param_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * A function over a dict element gets the dict whole, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testParamKeyed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_param_keyed", 0, 0);
  }

  /**
   * A mapped tuple is a tuple, no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair", 0, 0);
  }

  /**
   * The mapped tuple's first member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The mapped tuple's second member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testIds() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_ids", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A mapped flat element.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * A function picking a dict entry.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyedA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * A batch of a mapped tuple batches each member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testBatchedX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_batched_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A function with a defaulted parameter beyond the element's arity still gets the members.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testDefaultX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_default_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A variadic function gets exactly the element's members, so its last is the last member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testStarLast() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_star_last", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A function returning a list makes a tuple element, as {@code tf.data} packs it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testListA() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_list_a", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A function over either of two datasets, one with a pair element and one flat, gets either's
   * first argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMixedFirst() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_mixed_first",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4), TensorType.of(FLOAT_32, 5))));
  }

  /**
   * A function that replaces an entry of its dict argument changes its own dict, not the receiver's
   * element, so iterating the receiver still reads the entry's own type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUntouched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_untouched", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * A tuple element longer than the four positions the analysis spreads makes the mapped element
   * unknown, never the default the function returns when it is called with fewer.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFifth() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fifth", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * A member a function makes by an operation the analysis does not model holds nothing in the
   * analysis, which is not the same as being absent: it is a tensor of unknown type, not no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRawSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_raw_second", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * As {@link #testRawSecond()}, batched.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRawBatchedSecond() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE, "consume_raw_batched_second", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * The modeled member beside it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRawFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_raw_first", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * As {@link #testRawSecond()}, for a member of the structure a dataset is built from.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testRawSliceSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_raw_slice_second", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * A {@code None} member of the structure a dataset is built from is no tensor: its position holds
   * the {@code None} constant, not nothing, so it is not read as an unmodeled member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNoneAux() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_none_aux", 0, 0);
  }

  /**
   * As {@link #testNoneAux()}, for a {@code None} member a function returns, batched.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNoneMapped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_none_mapped", 0, 0);
  }
}
