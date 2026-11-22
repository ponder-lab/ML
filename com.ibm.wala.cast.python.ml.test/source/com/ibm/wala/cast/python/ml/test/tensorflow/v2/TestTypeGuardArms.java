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
   * object, so the conversion never runs and the result is the float64 argument alone at run time.
   *
   * <p>The test is left undecided, so the conversion's float32 also reaches the result: the front
   * end lowers {@code is} as {@code ==}, under which an equal dtype would decide the guard true.
   *
   * <p>TODO: Read float64 alone once the front end lowers an identity test as one
   * (https://github.com/wala/ML/issues/1078).
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

  /**
   * Every dtype is true, so {@code if x.dtype:} keeps a float32 array, and its int32 conversion is
   * dead.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTruthKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_truth_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A dtype equals the string naming it, so {@code x.dtype == "float32"} is not decided as a
   * comparison of two different values: the array that runs reaches the result, and so does its
   * int32 conversion.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testStringKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE, "consume_string_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_INT32)));
  }

  /**
   * {@code "float32" == x.dtype}, with the dtype on the right, is not decided either.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testStringLeftKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_string_left_kept",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_INT32)));
  }

  /**
   * NumPy orders dtypes by safe casting, so {@code x.dtype < np.float64} holds for a float32 array.
   * The comparison is not decided: the array that runs reaches the result, and so does its int32
   * conversion.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testOrderedKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_ordered_kept",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_INT32)));
  }

  /**
   * An instance of a program class extending {@code list} is a list, so {@code isinstance(x, list)}
   * is not decided false for it, and the conversion reaches the result.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubclassConverted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_subclass_converted", 1, 1, Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * An argument that may be a list or an array decides {@code isinstance(x, list)} for neither, so
   * both the conversion and the array reach the result.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEitherConverted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_either_converted",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32, TENSOR_2_3_INT32, TENSOR_2_3_FLOAT32)));
  }

  /**
   * A loop's guard reads {@code x} through the merge at the loop's head, whose arm from the body
   * the guard itself decides. The loop's value is no tensor, so an array is kept alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLoopKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_loop_kept", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A merge of a float32 and a float64 array has no one dtype, so its test against {@code
   * np.float64} is not decided: both arrays reach the result, and so does the int32 conversion that
   * runs for the float64 one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testArmsDisagree()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_arms_disagree",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_FLOAT64, TENSOR_2_3_INT32)));
  }

  /**
   * A value defined on one arm alone reads the null constant on the other, which has no dtype, so
   * the guard is not decided: the array that runs reaches the result, and so does its int32
   * conversion.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testOneArmedKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_one_armed_kept",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_INT32)));
  }

  /**
   * A merge of two float32 arrays has their one dtype, so its test against {@code np.float64} is
   * decided, and the int32 conversion is dead.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testArmsAgree()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_arms_agree", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A call nothing models has no target, so a guard on its value is not decided: the array that
   * runs reaches the result, and so does its int32 conversion.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnknownCallKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_unknown_call_kept",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_INT32)));
  }

  /**
   * The value of a call nothing models is no object the analysis holds, so {@code isinstance} over
   * it is not decided: the array that runs reaches the result, and so does its int32 conversion.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnknownObjectKept()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_unknown_object_kept",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_2_3_INT32)));
  }

  /**
   * A merge one of whose arms is the value of a call nothing models, a list at run time, declines
   * {@code isinstance(y, list)} rather than reading the verdict from the array alone, so the
   * conversion the list takes reaches the result beside the array.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnpickledConverted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_unpickled_converted",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_INT32, TENSOR_2_3_FLOAT32)));
  }
}
