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
   * The repeated-None idiom: the dead arm's unknown-dtype twin is gone (wala/ML#961), and so is the
   * rankless float32 member the dead arm's concat used to add (wala/ML#962). That concat's element
   * is the unstack over the None-only {@code past}, an operation that cannot execute; the element
   * read finds it through its producer and the concat is no tensor on both axes, so no feed refills
   * it and the live arm's shape stands alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRepeatedNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_repeated", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
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
    test(FILE, "consume_direct", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * The control for the element rule (wala/ML#962): a concat whose other element is a real Python
   * list beside the tensor is feasible, so the rule stays silent and the result keeps its rankless
   * float32 member. The rule reaches only an element produced by an operation that cannot execute
   * on its None-only input, never an element the reader merely could not size.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testListElementControl()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILE,
        "consume_list_element_control",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
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

  /**
   * The recursive case (wala/ML#962): the dead arm's outer concat takes an inner concat as its
   * element, and the inner concat's element is the unstack over the None-only {@code past}. The
   * element read follows the outer element to the inner concat and through it to the unstack, so
   * the outer concat is no tensor and the result reads the live arm alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedConcatOverNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_nested", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * The infeasible piece as the second element (wala/ML#962): the shape read's first element is the
   * live tensor, so the rule must reach the piece through the per-element read that follows, and
   * the concat is no tensor all the same.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSecondElementOverNone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_second", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * The quantifier control (wala/ML#962): the element's set mixes the dead arm's infeasible piece
   * with a live tensor, so the element may be live and the concat executes. The rule is universal
   * over the set, as the None-only read is, and the concat keeps its concatenated shape; an
   * existential reading would claim it cannot execute and drop it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMixedElementStaysTyped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_mixed", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6, 4))));
  }
}
