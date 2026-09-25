package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A tensor-input read whose points-to set is exactly the None constant reads no tensor
 * (wala/ML#961): the arm {@code if past is not None:} is dead when {@code past} is None, so the
 * unstack of {@code past} and the concat over its pieces contribute nothing. Each idiom reaches the
 * read with different evidence: {@code [None] * n} through wala/ML#960's elements, a direct {@code
 * past=None} through the call's own constant. A driver passing a real tensor keeps its members.
 */
public class TestNoneInput extends AbstractTensorTest {

  private static final String FILE = "tf2_test_none_input.py";

  /**
   * The repeated-None idiom: the dead arm's unknown-dtype twin is gone. The rankless float32 member
   * beside the real shape is the dead arm's concat, seeded no-tensor but refilled by its feed from
   * the live operand (an empty seed reads as unresolved to the feed loop); it widens the shape only
   * and is a separate item.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepeatedNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_repeated",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The direct {@code past=None} idiom reads the same as the repeated one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDirectNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_direct",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A dead arm whose concat takes {@code past} itself as an element: the element read applies the
   * rule, the concat is no tensor, and nothing refills it, so the result reads the live arm alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDirectNoneConcatElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_direct_concat", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * The control: a driver passing a real {@code past} keeps the live arm's concatenated shape
   * beside the plain one, since the rule is per context and an empty or typed set is untouched.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTensorPastKeepsItsMembers()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_with_past",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32, TensorType.of(FLOAT_32, 2, 8, 4))));
  }
}
