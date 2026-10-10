package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that an element of a tensor's static shape, {@code t.shape[i]}, is a Python scalar: a list
 * literal of such elements converts to a vector as long as the list, and a tensor divided by it
 * keeps its own shape.
 */
public class TestStaticShapeElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_cast_shape_list.py";

  /**
   * The cast of a four-element list of static shape elements: a float32 vector of four.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCastOfShapeElements()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_cast", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A {@code (1, 4)} tensor divided by that vector: {@code (1, 4)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDividedByShapeElements()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_divided", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 4))));
  }

  /**
   * A stack of two static shape elements: an int32 vector of two, as the packed elements are
   * scalars.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testStackOfShapeElements()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stacked", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }
}
