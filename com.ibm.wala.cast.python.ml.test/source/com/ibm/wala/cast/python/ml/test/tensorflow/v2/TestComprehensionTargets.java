package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Comprehensions whose target destructures, and comprehensions and numpy calls shared by several
 * callers (<a href="https://github.com/wala/ML/issues/955">wala/ML#955</a>). The fixture asserts
 * each consumer's type at run time.
 */
public class TestComprehensionTargets extends AbstractTensorTest {

  private static final String FILE = "tf2_test_comprehension_destructure.py";

  /**
   * A comprehension over {@code enumerate} whose target is {@code i, w} binds {@code w}: the
   * element {@code w + 1.0} is the {@code (2, 3)} float32 tensor it is at run time. The
   * comprehension trampoline passes one value per iterable, and the front end had flattened the
   * target into two formals, so {@code w} received nothing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDestructuringTarget()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A comprehension inside a per-caller method keeps each caller's element type: {@code predict}
   * called with a float32 list reads float32 alone, where one shared comprehension node had joined
   * the int32 caller's element in.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testComprehensionPerCallerFloat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * The int32 caller of {@link #testComprehensionPerCallerFloat()}'s helper reads int32 alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testComprehensionPerCallerInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "h", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * A numpy call inside a helper keeps each caller's dtype: {@code np.array} over a list of Python
   * ints reads int64 alone, where one shared {@code np.array} node had joined the float caller's
   * float64 in.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumpyPerCallerInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "k", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2))));
  }

  /**
   * The float caller of {@link #testNumpyPerCallerInt()}'s helper reads float64 alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumpyPerCallerFloat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "m", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 2))));
  }
}
