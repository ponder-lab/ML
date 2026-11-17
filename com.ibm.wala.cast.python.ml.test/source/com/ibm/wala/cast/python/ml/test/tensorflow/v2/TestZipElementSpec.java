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
 * Tests the {@code element_spec} of a {@code tf.data.Dataset.zip} that has an argument whose
 * element is not mirrored, a {@code map} over a file's lines, on {@code
 * tf2_test_zip_element_spec.py} (<a
 * href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>). A tensor made from a spec's
 * shape or dtype has the shape or dtype of the zip's element at the spec's place. Before, the spec
 * of such an argument's place held nothing. A tensor made from it where it was read was typed all
 * the same, from the read itself, so most of these pin that the description takes nothing away, and
 * the tests of a spec stored in an attribute are the ones that had no shape. {@link
 * com.ibm.wala.cast.python.ml.test.TestZipSpecProducers} pins the descriptions themselves.
 */
public class TestZipElementSpec extends AbstractTensorTest {

  private static final String FILE = "tf2_test_zip_element_spec.py";

  /** The specs of a zip, of its member and of the member's value are not tensors. */
  @Test
  public void testSpecIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_spec", 0, 0);
    test(FILE, "consume_member_spec", 0, 0);
    test(FILE, "consume_leaf_spec", 0, 0);
    test(FILE, "consume_leaf_shape", 0, 0);
  }

  /** The shapes a program maps out of the specs are not tensors. */
  @Test
  public void testMappedShapesAreNotTensors()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_shapes", 0, 0);
    test(FILE, "consume_shapes_member", 0, 0);
    test(FILE, "consume_shape_leaf", 0, 0);
  }

  /** The spec of one value of each member reads that value's shape and dtype alone. */
  @Test
  public void testMember()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_leaf_shape", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
    test(FILE, "use_other_leaf_shape", 1, 1, Map.of(2, Set.of(TENSOR_7_FLOAT32)));
    test(FILE, "use_leaf_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A control: a row of the zip reads the element, as before. */
  @Test
  public void testRow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_row", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 7))));
  }

  /** A spec stored in an attribute and read back reads the same there. */
  @Test
  public void testHeld()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_held_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 23))));
  }

  /** A zip of two functions' results that are each one tensor. */
  @Test
  public void testTensorResults()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_tensor_spec", 0, 0);
    test(FILE, "use_tensor_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 11))));
  }

  /**
   * A zip of a dataset whose element is mirrored, a function's results and a file's lines: each
   * member's spec reads its own member of the element. The mirrored member is described from both
   * of the attribute's sources, and the two descriptions read one type.
   */
  @Test
  public void testMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_mixed_mirrored_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 19))));
    test(FILE, "use_mixed_mapped_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 29))));
    test(FILE, "use_mixed_root_dtype", 1, 1, Map.of(2, Set.of(TensorType.of(STRING, 2))));
  }

  /** A zip in a zip, and a zip of a dict of datasets. */
  @Test
  public void testNesting()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_nested_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 31))));
    test(FILE, "use_keyed_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 37))));
  }

  /** The spec of a dataset that hands a zip's element on, a {@code shuffle}. */
  @Test
  public void testPassThrough()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_shuffled_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 41))));
  }

  /** The spec of a batch of a zip has the batch axis, so the member's size is the second. */
  @Test
  public void testBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_batched_leaf_shape", 0, 0);
    test(FILE, "use_batch_member_size", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 43, 2))));
  }

  /** The same for a spec stored in an attribute, which is read at the batch's own element. */
  @Test
  public void testHeldBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_held_batch_size", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 47, 2))));
  }

  /**
   * A {@code padded_batch} given the shapes mapped out of the specs as {@code padded_shapes} reads
   * the element the default padding reads: a batch of two whose last batch may be smaller.
   */
  @Test
  public void testPaddedBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<TensorType> padded = batchesOf(53);
    test(FILE, "use_padded", 1, 1, Map.of(2, padded));
    test(FILE, "use_padded_default", 1, 1, Map.of(2, padded));
  }

  /**
   * The same in a function a lambda handed to {@code apply} passes its dataset to: the shapes
   * mapped out of the specs there are not tensors, and the batch padded to them reads the element.
   */
  @Test
  public void testApplied()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_applied_shapes_member", 0, 0);
    test(FILE, "consume_applied_shape_leaf", 0, 0);
    test(FILE, "use_applied_row", 1, 1, Map.of(2, batchesOf(59)));
  }

  /**
   * The types of a batch of two, whose last batch may be smaller, of {@code int32} rows.
   *
   * @param size The size of a row.
   * @return The two types.
   */
  private static Set<TensorType> batchesOf(int size) {
    return Set.of(
        TensorType.of(INT_32, 2, size),
        new TensorType(INT_32, List.of(new SymbolicDim("?"), new NumericDim(size))));
  }
}
