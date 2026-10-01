package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a starred element in a tuple or list literal unpacks its iterable's elements into the
 * literal (<a href="https://github.com/wala/ML/issues/989">wala/ML#989</a>), where the iterable
 * stood for a single element and every later index read the wrong value.
 */
public class TestStarredLiteral extends AbstractTensorTest {

  private static final String FILE = "tf2_test_starred_literal.py";

  /**
   * An index into a literal with a starred element reads one of the literal's elements, never the
   * unpacked list itself: {@code (5, *head)[1]} reads {@code 5}, {@code 3} or {@code 4}, since the
   * unpacked length, and so every element's position, is not known; at run time it is {@code 3}.
   * Before, the list {@code [3, 4]} stood for one element and the read gave the shape {@code (3,
   * 4)}.
   */
  @Test
  public void testIndexAfterStar() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_index",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_64, 5),
                TensorType.of(FLOAT_64, 3),
                TensorType.of(FLOAT_64, 4))));
  }

  /** Iterating the literal yields the plain element and every unpacked element. */
  @Test
  public void testIteration() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_element",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 4), TensorType.of(FLOAT_32, 2), TensorType.of(INT_32, 3))));
  }

  /**
   * A literal with a starred element as a shape argument has an unknown length, so the reshaped
   * tensor's rank is unknown rather than read from the literal's writes: {@code (*rest, 3)} with
   * {@code rest} of length one is {@code (4, 3)} at run time, but the analysis cannot know the
   * unpacked length. Before, the unpacked list stood for one dimension and gave rank two whatever
   * its length; reading the literal's elements as unpositioned without this answered a scalar.
   */
  @Test
  public void testStarredShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_reshaped", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /** A literal starred in place, {@code [a, *[b, g]]}, unpacks its elements. */
  @Test
  public void testInlineStarredLiteral()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_inline",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 4), TensorType.of(FLOAT_32, 2), TensorType.of(INT_32, 3))));
  }

  /**
   * A starred iterable that is not a list or tuple, here an ndarray, is kept as itself, as the
   * literal kept it before: its elements are not fields the analysis knows. At run time the
   * elements are the array's rows, of shape {@code (4,)}.
   */
  @Test
  public void testNonCollectionStarKeptWhole()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_array_star",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4), TensorType.of(FLOAT_64, 2, 4))));
  }
}
