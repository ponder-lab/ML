package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a helper converting its argument only when the argument has another type returns, for
 * an argument whose type decides the guard, the value of the arm that runs alone (wala/ML#1053).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestTypeGuardArms extends AbstractTensorTest {

  private static final String FILE = "tf2_test_type_guard_arms.py";

  /**
   * A float32 array is not converted by {@code if x.dtype == np.float64:}, so the conversion's
   * int32 does not reach the result.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFloat32Kept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_float32_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A float64 array is converted by {@code if x.dtype == np.float64:}, so the unconverted argument
   * does not reach the result beside the conversion at run time.
   *
   * <p>The guard is not decided, so the unconverted argument also reaches the result: the front end
   * lowers {@code is} as {@code ==}, and an equal dtype decides {@code ==} but not {@code is},
   * which compares a dtype with a type object.
   *
   * <p>TODO: Read float32 alone once the front end lowers an identity test as one
   * (https://github.com/wala/ML/issues/1078).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFloat64Converted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_float64_converted",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_FLOAT64)));
  }

  /**
   * The guard {@code x.dtype != np.float64} decides as its negation: a float32 array returns
   * through the first return alone, and not as the second return's int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFloat32Unequal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_float32_unequal", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A float32 tensor is not cast to int32 by {@code if x.dtype == tf.float64:}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTensorKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_tensor_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A float64 array's dtype equals the TensorFlow token {@code tf.float64}, so the array is cast at
   * run time.
   *
   * <p>The guard is not decided, so the unconverted argument also reaches the result: the front end
   * lowers {@code is} as {@code ==}, and an equal dtype decides {@code ==} but not {@code is},
   * which compares a dtype with a type object.
   *
   * <p>TODO: Read float32 alone once the front end lowers an identity test as one
   * (https://github.com/wala/ML/issues/1078).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTensorMixedToken()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_tensor_mixed_token",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_FLOAT64)));
  }

  /**
   * An array is not a list, so {@code if isinstance(x, list):} does not convert it to int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testArrayKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_array_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A list is converted by {@code if isinstance(x, list):}, so the list itself, no tensor, does not
   * reach the result beside the array.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testListConverted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_list_converted", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * The issue's helper over a float32 array: neither conversion runs.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCastXKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_cast_x_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The issue's helper over a float64 array: the second conversion runs, and the argument does not
   * reach the result beside it at run time.
   *
   * <p>The guard is not decided, so the unconverted argument also reaches the result: the front end
   * lowers {@code is} as {@code ==}, and an equal dtype decides {@code ==} but not {@code is},
   * which compares a dtype with a type object.
   *
   * <p>TODO: Read float32 alone once the front end lowers an identity test as one
   * (https://github.com/wala/ML/issues/1078).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCastXConverted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_cast_x_converted",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_FLOAT64)));
  }

  /**
   * The issue's helper with conversions to other dtypes, over a float32 array: the first guard is
   * decided by the array's type, so the second reads the argument's dtype alone through the first
   * guard's merge, and neither conversion reaches the result.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCastXChained()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_cast_x_chained", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * {@code x.dtype is np.float64} compares a dtype with a type object, which are never the same
   * object, so the guard is not decided as {@code ==} would be. The test is left undecided: both
   * the float64 argument and the conversion reach the result.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIdentityUndecided()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_identity_kept",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT64, TENSOR_2_3_FLOAT32)));
  }
}
