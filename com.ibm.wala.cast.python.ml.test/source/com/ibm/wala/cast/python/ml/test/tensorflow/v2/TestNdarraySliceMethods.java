package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A slice of an array, and the result of arithmetic on one, is an array with its own allocation
 * that keeps the array's methods, so {@code astype} and {@code tolist} on it dispatch (<a
 * href="https://github.com/wala/ML/issues/1009">wala/ML #1009</a>). The slice {@code x[1:3]}'s
 * leading extent stays unresolved: the slice generator's plan for a bare slice does not fold its
 * bounds, which this test does not exercise. wala/ML#551, which moves the methods to the class,
 * must keep these green.
 */
public class TestNdarraySliceMethods extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_ndarray_slice_methods.py";

  /**
   * A slice's {@code astype} dispatches and narrows the dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSliceAstype() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_astype",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(INT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(3))))));
  }

  /**
   * A slice's {@code tolist} dispatches and keeps the dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSliceTolist() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_tolist",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(3))))));
  }

  /**
   * An arithmetic result's {@code astype} dispatches.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testArithmeticAstype() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_scaled", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 4, 3))));
  }

  /**
   * An {@code ndarray.reshape} result read through a tuple's element, as {@code x, y =
   * a.reshape(...), b.reshape(...)} does, is typed by the reshape: the value reaches the array the
   * method's body allocates, and the reshape generator answers for it as it does for the call's own
   * result (wala/ML#1009).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testUnpackedReshape() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ndarray_reshape_unpacked.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 256, 784), TensorType.of(FLOAT_32, 96, 784))));
  }

  /**
   * Arithmetic on an array a Keras dataset loader returns is an array (wala/ML#1009): {@code
   * mnist.load_data()} allocates its arrays as classes of their own, and the arithmetic result was
   * allocated only for the array, tensor and variable types, so {@code x_train / 255.0} had no
   * value, and the expanded images reached the dataset as nothing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDatasetLoaderArrayArithmetic()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ndarray_newaxis_dataset.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 32, 28, 28, 1))));
  }
}
