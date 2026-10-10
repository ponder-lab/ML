package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
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

  /**
   * A separate integer the analysis cannot read, {@code reshape(rows, 3)} with {@code rows} parsed
   * at run time: the rank and the readable size are kept and the unread size is unresolved, never
   * the receiver's shape.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnreadSeparateDimension()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unread",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(INT_64, List.of(UnresolvedDim.INSTANCE, new NumericDim(3))))));
  }

  /**
   * The control: the dimensions spread from a list, {@code reshape(*dims)}, whose positions past
   * the star cannot be aligned, keep the reading the one-argument form already gives them, {@code
   * (2, 3)}, as the spread list is read as the shape.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testStarredDimensions() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_starred", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }

  /**
   * A row of one helper's {@code a.reshape(n, m)} called with {@code (2, 3)}, beside a call with
   * {@code (3, 2)}: each call's dimensions are its own, {@code (3,)} here.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHelperRowsOfTwoByThree()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_helper_rows_of_two_by_three",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 3))));
  }

  /**
   * The other call of that helper, {@code (3, 2)}: rows of {@code (2,)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHelperRowsOfThreeByTwo()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_helper_rows_of_three_by_two",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 2))));
  }

  /**
   * The reshaped array read back out of a list, so its type comes from the reshape's own allocation
   * rather than from the call: {@code (2, 3)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHeldReshape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_held", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }

  /**
   * A list-held reshape built in a helper called with {@code (2, 3)}, beside a call with {@code (3,
   * 2)}, read back out of the returned list: the dimensions are read at each call, {@code (2, 3)}
   * here.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHeldTwoByThree() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_held_two_by_three", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }

  /**
   * The other call of that list-holding helper, {@code (3, 2)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHeldThreeByTwo() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_held_three_by_two", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3, 2))));
  }

  /**
   * A loss test's arrays, each a flat literal reshaped by separate integers bound to locals, {@code
   * reshape(B, T, U, V)}: {@code (2, 4, 3, 3)}, beside a {@code (2,)} array built directly.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLossArrays() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "LossTest._run",
        3,
        3,
        Map.of(
            3,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 3)),
            4,
            Set.of(TensorType.of(FLOAT_32, 2)),
            5,
            Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 3))));
  }

  /**
   * The reshaped array unpacked from a tuple a helper returns, called with {@code (2, 3)} beside a
   * call with {@code (3, 2)}: the array reaches the reshape through the tuple's element, so its
   * dimensions are read through the reshape's callers, each in its own frame.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPairedTwoByThree() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_paired_two_by_three", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 3))));
  }

  /**
   * The other call of that tuple-returning helper, {@code (3, 2)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPairedThreeByTwo() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_paired_three_by_two", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3, 2))));
  }
}
