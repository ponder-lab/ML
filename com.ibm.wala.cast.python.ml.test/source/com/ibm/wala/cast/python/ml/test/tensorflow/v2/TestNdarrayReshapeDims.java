package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests an array's {@code reshape} given its dimensions as separate integers, {@code x.reshape(2,
 * 3)}, which NumPy reads as the shape {@code (2, 3)} just as it reads the tuple {@code (2, 3)}
 * (wala/ML#1034). Only the first integer was read as the target shape, so the array read as {@code
 * (2,)}, and its rows and subscripts as scalars.
 */
public class TestNdarrayReshapeDims extends AbstractTensorTest {

  private static final String FILE = "tf2_test_ndarray_reshape_dims.py";

  /**
   * An array reshaped by two separate integers, {@code reshape(2, 3)}: {@code (2, 3)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSeparateDimensions()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_two", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }

  /**
   * An array reshaped by three separate integers, {@code reshape(1, 2, 3)}: {@code (1, 2, 3)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testThreeSeparateDimensions()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_three", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 1, 2, 3))));
  }

  /**
   * A row of an array reshaped by separate integers, bound by iterating it: {@code (3,)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testRowOfSeparateDimensions()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_row", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3))));
  }

  /**
   * An integer subscript of an array reshaped by separate integers: {@code (3,)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSubscriptOfSeparateDimensions()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_subscript", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3))));
  }

  /**
   * Separate integers with an inferred {@code -1}, {@code reshape(-1, 2)} of six elements: {@code
   * (3, 2)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testInferredSeparateDimension()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inferred", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3, 2))));
  }

  /**
   * The control: one integer, {@code reshape(6)}, already read as {@code (6,)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSingleDimension() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_one", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 6))));
  }

  /**
   * The control: one tuple, {@code reshape((2, 3))}, already read as {@code (2, 3)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTupleDimensions() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tuple", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }
}
