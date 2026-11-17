package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a dataset's {@code element_spec} describes the dataset's element and is not the
 * element, on {@code tf2_test_shape_producers.py} (<a
 * href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>): the spec, its members and its
 * shape are not tensors, and a tensor made from the spec's shape or dtype has the element's.
 * Before, the read was typed as the dataset's element, so a spec was a tensor, and its {@code
 * shape} was not read.
 */
public class TestElementSpec extends AbstractTensorTest {

  private static final String FILE = "tf2_test_shape_producers.py";

  /** A spec is not a tensor, and neither is a tuple of specs or its member. */
  @Test
  public void testSpecIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_spec", 0, 0);
    test(FILE, "consume_tuple_spec", 0, 0);
    test(FILE, "consume_tuple_spec_member", 0, 0);
  }

  /** A spec's shape is not a tensor, wherever in the element's structure the spec is. */
  @Test
  public void testSpecShapeIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_spec_shape", 0, 0);
    test(FILE, "consume_tuple_spec_shape", 0, 0);
    test(FILE, "consume_dict_spec_shape", 0, 0);
    test(FILE, "consume_nested_spec_shape", 0, 0);
    test(FILE, "consume_named_spec_shape", 0, 0);
    test(FILE, "consume_parameter_spec_shape", 0, 0);
    test(FILE, "consume_map_spec_shape", 0, 0);
    test(FILE, "consume_spec_batch_size", 0, 0);
  }

  /**
   * The spec of a dataset whose element is not mirrored, a dataset of a file's lines, is not a
   * tensor, and neither is its shape. It is the read's receiver, a dataset, that tells the read
   * from a read of the element, so this holds whether or not the element is described.
   */
  @Test
  public void testSpecOfUnmirroredElementIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_text_spec", 0, 0);
    test(FILE, "consume_text_spec_shape", 0, 0);
    test(FILE, "consume_text_map_spec_shape", 0, 0);
  }

  /** The spec of a dataset held by a variable that may be {@code None} is not a tensor. */
  @Test
  public void testSpecOfOptionalDatasetIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_optional_spec", 0, 0);
  }

  /** A subscript of the spec's shape, and of the list its {@code as_list()} returns, as a size. */
  @Test
  public void testSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_spec_size", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
    test(FILE, "use_spec_list_size", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /** The spec's shape as the shape argument of an allocator. */
  @Test
  public void testAllocator()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_unbatched_spec_shape", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * The spec's shape as a list, sliced and concatenated, as the shape argument of {@code
   * tf.reshape}, beside a control: the same list built from a tensor's shape.
   */
  @Test
  public void testReshape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_spec_reshape", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
    test(FILE, "use_sliced_list_reshape", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The same list as the shape argument of an allocator, which does not read it, whether it is
   * built from the spec's shape or from a tensor's.
   */
  // TODO: Expect `TENSOR_2_3_FLOAT32` for both once an allocator reads a shape passed as a list
  // built from `as_list()` (wala/ML#1285).
  @Test
  public void testAllocatorOfSlicedList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_spec_shape", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
    test(FILE, "use_sliced_list", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** The spec's dtype as an allocator's, beside a control: a tensor's. */
  @Test
  public void testDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_spec_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test(FILE, "use_tensor_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * The shape and the dtype of a spec passed to a function, where the read's receiver is a
   * parameter and the dataset the spec describes is out of sight.
   */
  @Test
  public void testPassedSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_passed_spec_shape", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
    test(FILE, "use_passed_spec_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * The shape and the dtype of a spec that reaches a function as a leaf of the structure of specs
   * {@code tf.nest.map_structure} maps, where the read of the attribute is the structure's.
   */
  @Test
  public void testMappedSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_mapped_spec_shape", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
    test(FILE, "use_mapped_spec_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * The shape of the spec of one value of a dict whose values differ in shape, passed to a
   * function: the described value's, and not every value's.
   */
  @Test
  public void testMemberSpecShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_member_spec_shape", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** The dtype of the spec of one value of a dict whose values differ in dtype, passed on. */
  @Test
  public void testMemberSpecDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_member_spec_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * A control: a member of a dataset's element that is named as the attribute is, read off the
   * element, is the element's tensor.
   */
  @Test
  public void testElementMemberNamedAsTheAttribute()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_keyed_element", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * The same control for a dataset whose element is not mirrored, where a {@code map} function's
   * parameter may be the dataset itself.
   */
  @Test
  public void testMemberOfUnmirroredElementNamedAsTheAttribute()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_keyed_row", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** The shape of a spec a program constructs, as the shape argument of an allocator. */
  @Test
  public void testConstructedSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_constructed_spec_shape", 0, 0);
    test(FILE, "use_constructed_spec_shape", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
