package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code np.expand_dims}, {@code np.squeeze} and {@code np.clip} (wala/ML#1308): each returns a new
 * array whose type comes from its argument's, so the array is typed and so is a tensor made of it.
 */
public class TestNumpyExpandDimsSqueezeClip extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_numpy_expand_squeeze_clip.py";

  /** The project whose sidecar alone types the arrays its loaders unpickle. */
  private static final String PROJECT = "sidecar_numpy_clip_proj";

  private static final String[] FILES = {PROJECT + "/loader.py", PROJECT + "/driver.py"};

  /**
   * {@code np.expand_dims} with a constant axis, from the start, from the end and by keyword: the
   * argument's shape with a size-1 axis at that position, and the argument's dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testExpandDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "expanded", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 2, 4))));
    test(FIXTURE, "expanded_at_the_end", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 1))));
    test(FIXTURE, "expanded_by_keyword", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 1, 4))));
  }

  /**
   * {@code np.expand_dims} with a tuple of axes: the shape is unknown, and the dtype is the
   * argument's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testExpandDimsByAxes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "expanded_by_axes", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A Python number, list or tuple given to any of the three functions, a number that the {@code
   * float} builtin returns among them: the shape is the one the function gives the converted array,
   * and the dtype is unknown, since the argument is read as TensorFlow converts it and NumPy
   * converts it otherwise.
   *
   * <p>TODO: Each reads its NumPy dtype once wala/ML#1309 is resolved: {@code int64} for {@code
   * expanded_list}, {@code clipped_list} and {@code clipped_number}, and {@code float64} for {@code
   * expanded_number}, {@code squeezed_list} and {@code clipped_tuple}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPythonInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "expanded_list", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 1, 3))));
    test(FIXTURE, "expanded_number", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 1))));
    test(FIXTURE, "squeezed_list", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 2))));
    test(FIXTURE, "clipped_list", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 3))));
    test(FIXTURE, "clipped_tuple", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 3))));
    test(FIXTURE, "clipped_number", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN))));
  }

  /**
   * {@code np.squeeze} with no axis, with a positional one and with a keyword one: the argument's
   * shape without the size-1 axes, or without the named one, and the argument's dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueeze()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "squeezed", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FIXTURE, "squeezed_by_axis", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 1))));
    test(FIXTURE, "squeezed_by_keyword", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 2))));
  }

  /**
   * {@code np.clip} with bounds that leave the array's dtype as it is: integers and {@code None}
   * beside a floating-point array, a float beside one, and integers an integral array's dtype
   * holds.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testClipKeepsType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "clipped", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FIXTURE, "clipped_from_below", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FIXTURE, "clipped_by_keywords", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FIXTURE, "clipped_by_negative", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FIXTURE, "clipped_integers", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 2))));
    test(FIXTURE, "clipped_bytes", 1, 1, Map.of(2, Set.of(TensorType.of(UINT_8, 2, 2))));
  }

  /**
   * {@code np.clip} with bounds NumPy promotes the array with: a float beside an integral array
   * gives {@code float64}, and a number the array's dtype does not hold gives an unknown dtype. The
   * shape is the array's in each.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testClipPromotes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FIXTURE,
        "clipped_integers_by_floats",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 2, 2))));
    test(
        FIXTURE,
        "clipped_bytes_by_negative",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UNKNOWN, 2, 2))));
    test(FIXTURE, "clipped_by_wide_integer", 1, 1, Map.of(2, Set.of(TensorType.of(UNKNOWN, 2, 4))));
  }

  /**
   * {@code np.clip} with a bound that is an array: the bound may broadcast the result to a shape
   * that is not the clipped array's, and its dtype may be the result's, so both are unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testClipByArray()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "clipped_by_array", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A tensor made of each function's result has the result's type: the functions allocate their own
   * arrays, so the tensor's producer finds the function and not its argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTensorOfResult()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FIXTURE, "variable_of_expanded", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 2, 4))));
    test(
        FIXTURE, "constant_of_expanded", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 2, 4))));
    test(FIXTURE, "variable_of_squeezed", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FIXTURE, "variable_of_clipped", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * {@code np.clip} of a value that only the sidecar types, beside bounds that leave every numeric
   * dtype as it is: the feed forwards the argument's shape and dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFedClip()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "fed_clip",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * {@code np.clip} of a value that only the sidecar types, beside a negative bound, which some
   * dtypes do not hold: the feed forwards the argument's shape alone, and the dtype is unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFedClipByNegative()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "fed_clip_by_negative",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UNKNOWN, 4, 3))));
  }

  /**
   * {@code np.clip} of a value that only the sidecar types, beside a bound that is an array: no
   * feed is declared, and the shape and the dtype are unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFedClipByArray()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "fed_clip_by_array",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * {@code np.expand_dims} of a value that only the sidecar types: the feed forwards the argument's
   * dtype, and the shape, which the feed does not transform, is unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFedExpandDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "fed_expanded",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * {@code np.squeeze} of a value that only the sidecar types: the feed forwards the argument's
   * dtype, and the shape is unknown.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFedSqueeze()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "fed_squeezed",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }
}
