package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of plain function dispatch and redefinition ({@code Function*}/{@code RedefinedFunction*}),
 * carved from the {@link TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestFunctions extends AbstractTensorTest {

  @Test
  public void testFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function2.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function3.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function4.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function5.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function6.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testFunction7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function7.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testFunction8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function8.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_2_FLOAT32)));
  }

  @Test
  public void testFunction9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function9.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testFunction10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function10.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_INT32)));
  }

  @Test
  public void testFunction11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function11.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_3_3_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/308. */
  @Test
  public void testFunction12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function12.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_3_2_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/308.
   *
   * <p>This one has lexical scoping.
   */
  @Test
  public void testFunction13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function13.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_3_2_FLOAT32)));
  }

  /**
   * The first definition of the redefined top-level function of <a
   * href="https://github.com/wala/ML/issues/719">wala/ML#719</a>: the script defines {@code
   * compute} twice and calls each definition. Definition-site-distinct synthetic classes keep both
   * bodies, the module binding rebinds at the second {@code def}, and straight-line SSA binds each
   * call to the definition in scope, so this definition's parameter carries exactly its own call's
   * {@code (2, 2)} — where the pre-fix collapse lost this body entirely and bound its call to the
   * second definition.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRedefinedFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_redefined_function.py",
        "compute",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }

  /**
   * The second definition of {@link #testRedefinedFunction()}'s redefined function (wala/ML#719):
   * the definition at line 11 composes the position-disambiguated synthetic class {@code
   * compute$11}, and its parameter carries exactly the second call's {@code (3, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRedefinedFunction2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_redefined_function.py",
        "compute$11",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }
}
