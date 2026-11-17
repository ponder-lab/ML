package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a tensor's {@code get_shape()} is read as its {@code shape} attribute is, in the forms
 * a program consumes a shape in, on {@code tf2_test_shape_producers.py} (<a
 * href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>). Each form has a sink for the
 * attribute and one for the method, and both read the same type. Before, a shape taken from the
 * method was not read: a tensor allocated or reshaped with it had an unknown rank, and a size read
 * from it was unresolved.
 */
public class TestStaticShapeMethod extends AbstractTensorTest {

  private static final String FILE = "tf2_test_shape_producers.py";

  private static final TensorType TENSOR_7_7_FLOAT32 = TensorType.of(FLOAT_32, 7, 7);

  /** The shape argument of {@code tf.reshape}. */
  @Test
  public void testReshape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_reshape_attribute", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
    test(FILE, "use_reshape_method", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** The shape argument of an allocator. */
  @Test
  public void testAllocator()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_zeros_attribute", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
    test(FILE, "use_zeros_method", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** The shape argument of an allocator, as the list {@code as_list()} returns. */
  // TODO: Expect `TENSOR_4_3_FLOAT32` for both once an allocator reads a shape passed whole as a
  // list (wala/ML#1285).
  @Test
  public void testAllocatorOfList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_zeros_attribute_list", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
    test(FILE, "use_zeros_method_list", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** A subscript of the shape, as a size in a shape list. */
  @Test
  public void testSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_size_attribute", 1, 1, Map.of(2, Set.of(TENSOR_4_2_FLOAT32)));
    test(FILE, "use_size_method", 1, 1, Map.of(2, Set.of(TENSOR_4_2_FLOAT32)));
  }

  /** A subscript of the list {@code as_list()} returns, as a size in a shape list. */
  @Test
  public void testListSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_list_size_attribute", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
    test(FILE, "use_list_size_method", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /** The list {@code as_list()} returns, assigned to a local and subscripted, in a function. */
  @Test
  public void testAssignedList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_assigned_list_attribute", 1, 1, Map.of(2, Set.of(TENSOR_4_2_FLOAT32)));
    test(FILE, "use_assigned_list_method", 1, 1, Map.of(2, Set.of(TENSOR_4_2_FLOAT32)));
  }

  /**
   * A rank guard, {@code if t.get_shape().ndims == 2}, is decided as {@code if t.shape.ndims == 2}
   * is, so only the taken arm's tensor is returned. Undecided, the other arm's {@code (6, 6)} is
   * read beside it.
   */
  @Test
  public void testRankGuard()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_rank_attribute", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
    test(FILE, "use_rank_method", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /** The shape of a function's parameter. */
  @Test
  public void testParameter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_parameter_attribute", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
    test(FILE, "use_parameter_method", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * A control: a {@code get_shape} method of a program's own class is not the tensor's. The class
   * holds a {@code (4, 3)} tensor and its method returns {@code [7, 7]}, which is what the
   * allocator reads.
   */
  @Test
  public void testOwnMethod()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_own_method", 1, 1, Map.of(2, Set.of(TENSOR_7_7_FLOAT32)));
  }

  /**
   * A receiver that is a tensor at one iteration and an instance of a program's class at the next,
   * whose {@code get_shape} method, or {@code shape} attribute, is the list {@code [7, 7]}. The
   * allocator reads the list alone, in either form.
   */
  // TODO: Expect `TENSOR_4_3_FLOAT32` beside `TENSOR_7_7_FLOAT32` for both once an allocator reads
  // every allocation its shape argument holds (wala/ML#1286).
  @Test
  public void testEitherReceiver()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_either_attribute", 1, 1, Map.of(2, Set.of(TENSOR_7_7_FLOAT32)));
    test(FILE, "use_either_method", 1, 1, Map.of(2, Set.of(TENSOR_7_7_FLOAT32)));
  }

  /**
   * A control: the same receiver, where the program's {@code get_shape} returns a list that is not
   * read. The call has the tensor's method and the program's as targets, so it is not read as the
   * tensor's shape, and the rank is unknown. Read as the tensor's shape, it would be {@code (4, 3)}
   * alone, without the program's {@code (7, 7)}.
   */
  @Test
  public void testOpaqueMethod()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_opaque_method", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** The shape a method returns is not a tensor, and neither is a size read from it. */
  @Test
  public void testShapeIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_method", 0, 0);
    test(FILE, "consume_variable_method", 0, 0);
    test(FILE, "consume_method_list", 0, 0);
    test(FILE, "use_dimension_method", 0, 0);
  }
}
