package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a slice of a list or tuple literal by constant bounds holds only the elements in range
 * (<a href="https://github.com/wala/ML/issues/993">wala/ML#993</a>), where the slice aliased its
 * receiver and carried the elements it drops. Each is observed through an operation that reads the
 * slice's elements, as a layer's split of its input does.
 */
public class TestListSlice extends AbstractTensorTest {

  private static final String FILE = "tf2_test_list_slice.py";

  private static final TensorType INT_4_1 = TensorType.of(INT_32, 4, 1);

  /**
   * A list slice with a bound counted from the end drops the trailing element: the concatenation of
   * what remains is the two float32 rows, not a mix with the trailing int32 column.
   */
  @Test
  public void testListDropsTrailing() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_list", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 3))));
  }

  /** A tuple slice with an absent lower bound keeps the leading elements. */
  @Test
  public void testTupleKeepsLeading() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 3))));
  }

  /** A slice with an absent upper bound keeps only the tail. */
  @Test
  public void testTail() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tail", 1, 1, Map.of(2, Set.of(INT_4_1)));
  }

  /**
   * A list that grows after it is built keeps its literal's elements past the upper bound, since a
   * bound counted from the end no longer names them: {@code [0:-1]} of {@code [a, h]} plus an
   * appended element is {@code [a, h]}, whose concatenation has six rows. Trimming by the literal's
   * length alone would keep only {@code a}.
   */
  @Test
  public void testGrownListStaysSound()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_appended", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6, 3))));
  }
}
