package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the {@code element_spec} of a dataset whose element is not mirrored, on {@code
 * tf2_test_unmirrored_element_spec.py} (<a
 * href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>): a dataset of a file's lines or
 * records or of file names, a {@code map} over one, a dataset of a generator's values, and a
 * dataset made from any of them. A tensor made from a spec's shape or dtype has the shape or dtype
 * of the element of the dataset the spec was read off, at the spec's place in the element. Before,
 * the read held nothing, so such a tensor had no shape and the default dtype.
 */
public class TestUnmirroredElementSpec extends AbstractTensorTest {

  private static final String FILE = "tf2_test_unmirrored_element_spec.py";

  /** A batch of two whose last batch may be smaller, of rows of three. */
  private static final Set<TensorType> PADDED =
      Set.of(
          TENSOR_2_3_FLOAT32,
          new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(3))));

  /** A dict of specs and its member are not tensors. */
  @Test
  public void testSpecIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_spec", 0, 0);
    test(FILE, "consume_member_spec", 0, 0);
  }

  /** The spec of a file's lines: a scalar of strings. */
  @Test
  public void testLines()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_line_shape", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
    test(FILE, "use_line_dtype", 1, 1, Map.of(2, Set.of(TensorType.of(STRING, 2))));
  }

  /** The spec of a dataset that hands its receiver's element on, a {@code shuffle}. */
  @Test
  public void testPassThrough()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_shuffled_shape", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** The specs of a dataset of file names and of a dataset of records. */
  @Test
  public void testFileNamesAndRecords()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_file_name_dtype", 1, 1, Map.of(2, Set.of(TensorType.of(STRING, 2))));
    test(FILE, "use_record_dtype", 1, 1, Map.of(2, Set.of(TensorType.of(STRING, 2))));
  }

  /**
   * The spec of one value of a dict a {@code map} function returns, whose values differ in shape
   * and dtype, reads that value's shape and dtype alone.
   */
  @Test
  public void testMember()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_member_shape", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
    test(FILE, "use_member_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * The spec of a batch of a {@code map}'s results has the batch axis, so the sizes after it are
   * the value's own: the spec is read off the batch, not off the value the function returned.
   */
  @Test
  public void testBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_batch_member_size", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
    test(FILE, "use_batch_member_sizes", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
    test(FILE, "use_tensor_batch_size", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * The batch axis of the spec of a batch that may be partial reads what an element of the batch
   * has there, the batch's size and the uncounted size of the last batch, where at run time the
   * spec's axis has no size.
   */
  // TODO: Expect a first axis with no size, or no tensor, once the spec of a batch that may be
  // partial has no size for the batch axis (wala/ML#1280).
  @Test
  public void testBatchAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "use_batch_axis",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 5),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(5))))));
  }

  /** The spec of a {@code map} whose function returns one tensor. */
  @Test
  public void testTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_tensor_shape", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /** The spec of a member of a tuple a {@code map} function returns, also passed to a function. */
  @Test
  public void testTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_pair_shape", 1, 1, Map.of(2, Set.of(TENSOR_7_FLOAT32)));
    test(FILE, "use_pair_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test(FILE, "use_passed_shape", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /**
   * A spec stored in an attribute and read back describes the element of the dataset it was read
   * off: a {@code map}'s results, and a batch of them, whose spec has the batch axis.
   */
  @Test
  public void testHeld()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_held_shape", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
    test(FILE, "use_held_batch_size", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 11, 2))));
  }

  /** A spec read out of the leaves {@code tf.nest.flatten} returns for the specs. */
  @Test
  public void testFlattened()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_flat_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 9))));
  }

  /** The spec of a value of a dict in a tuple in a dict. */
  @Test
  public void testNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_nested_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 9, 2))));
  }

  /** The spec of a dataset of a generator's values, from the element the call declares. */
  @Test
  public void testGenerator()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_generated_shape", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
    test(FILE, "use_generated_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * A batch padded to the shapes a program maps out of the specs reads as a batch padded by default
   * does.
   */
  @Test
  public void testPaddedBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_padded", 1, 1, Map.of(2, PADDED));
    test(FILE, "use_padded_default", 1, 1, Map.of(2, PADDED));
  }
}
