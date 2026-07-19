package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.MNIST_INPUT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT32;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of tensor-type inference through decorated, static, class, and same-named methods, carved
 * from the {@code TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestDecoratedMethods extends AbstractTensorTest {

  /**
   * Same-name-class guard for <a href="https://github.com/wala/ML/issues/678">wala/ML#678</a>: two
   * sibling scripts each define a Keras subclass named {@code GenGPT2} (with a {@code
   * super(GenGPT2, self)} by-name reference in {@code __init__}, mirroring the NLPGNN subject);
   * this pins that the first script's class keeps its call-graph nodes and its {@code predict}
   * parameter types.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameA()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_proj/tf2_test_samename_a.py", "samename_proj/tf2_test_samename_b.py"
        },
        "tf2_test_samename_a.py",
        "GenGPT2.predict",
        "samename_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testSamenameA()}: the second script's same-named class keeps its
   * call-graph nodes and typing too (wala/ML#678).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameB()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_proj/tf2_test_samename_a.py", "samename_proj/tf2_test_samename_b.py"
        },
        "tf2_test_samename_b.py",
        "GenGPT2.predict",
        "samename_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_3_3_FLOAT32)));
  }

  /**
   * Deep variant of {@link #testSamenameA()} mirroring the wala/ML#678 subject's dispatch shape:
   * both sibling scripts pass their same-named model into one shared helper ({@code
   * helpers/samples.py}'s {@code sample_sequence}), whose nested closures capture {@code model}
   * lexically and dispatch {@code model.predict} from a frame reached by both scripts — the NLPGNN
   * {@code nlpgnn/sample/samples.py} structure the two-file fixture lacks. Dispatch survives the
   * whole chain (no wala/ML#678 node loss at this scale), but the closure bodies are
   * call-string-keyed, so one {@code step} node serves both lexical parents and each sink receives
   * the cross-sibling union rather than its own shape (runtime truth here: {@code (2, 2) float32}).
   *
   * <p>TODO: Expect exactly {@code (2, 2) float32} once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object, the closure analogue of wala/ML#679.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameDeepA()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_deep_proj/helpers/__init__.py",
          "samename_deep_proj/helpers/samples.py",
          "samename_deep_proj/tf2_test_samename_deep_a.py",
          "samename_deep_proj/tf2_test_samename_deep_b.py"
        },
        "tf2_test_samename_deep_a.py",
        "consume",
        "samename_deep_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testSamenameDeepA()} (wala/ML#678): runtime truth is {@code (3, 3)
   * float32}; the extra member is the wala/ML#685 cross-sibling closure union.
   *
   * <p>TODO: Expect exactly {@code (3, 3) float32} once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameDeepB()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_deep_proj/helpers/__init__.py",
          "samename_deep_proj/helpers/samples.py",
          "samename_deep_proj/tf2_test_samename_deep_a.py",
          "samename_deep_proj/tf2_test_samename_deep_b.py"
        },
        "tf2_test_samename_deep_b.py",
        "consume",
        "samename_deep_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * A {@code @tf.function} without {@code input_signature} creates {@code tf.constant(5)} and
   * passes it to {@code g} (the FUT), so {@code g}'s parameter is that scalar. At runtime {@code g}
   * receives {@code ()} int32 and Ariadne agrees: a positive guard that a value flowing through a
   * decorated body to a callee types the callee's parameter correctly.
   */
  @Test
  public void testDecoratedCallDepth()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorated_call_depth.py", "g", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * A {@code @tf.function(input_signature=[(None,) int32])} passes its parameter to {@code g} (the
   * FUT). What {@code g} receives depends on the execution mode, which a static analysis cannot
   * determine: traced (the default) the signature governs and {@code g} receives {@code (None,)}
   * int32; under {@code run_functions_eagerly} the signature is ignored and {@code g} receives the
   * call-site argument's {@code (3,)} int32. So the sound type of {@code g}'s parameter is the set
   * {@code {(None,), (3,)}} int32.
   *
   * <p>TODO: this pins the current behavior. Ariadne does not consume {@code input_signature}, so
   * it produces only the argument-derived {@code (3,)} element and misses the signature-derived
   * {@code (None,)} one; the sound result is the set {@code {(None,), (3,)}} int32. Tracked by <a
   * href="https://github.com/wala/ML/issues/638">wala/ML#638</a>.
   */
  @Test
  public void testDecoratedCallDepthInputSig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_decorated_call_depth_input_sig.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * Verifies a class-body PEP-526 annotated assignment with a value ({@code weight: tf.Tensor =
   * tf.ones([3, 4])}) assigns the class attribute and that reading it back ({@code y = C.weight})
   * recovers the {@code (3, 4) float32} type (<a
   * href="https://github.com/wala/ML/issues/579">wala/ML#579</a>). This guards the value-bearing
   * branch of {@code visitAnnAssign}: unlike an annotation-only declaration ({@code x: T}, which
   * declares the field but emits no member {@code put}), a value-bearing class field still emits
   * the {@code put}, so the attribute is typed.
   */
  @Test
  public void testClassAttrAnnAssign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_classattr_annassign.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_3_4_FLOAT32)));
  }

  @Test
  public void testStaticMethod() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod2() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method2.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod3() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method3.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod4() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method4.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod5() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method5.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod6() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method6.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod7() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method7.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod8() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method8.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod9() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method9.py",
        "MyClass.the_static_method",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod10() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method10.py",
        "MyClass.the_static_method",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_static_method11.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod12() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_static_method12.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod13() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method13.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testStaticMethod14() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method14.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testClassMethod() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_class_method.py",
        "MyClass.the_class_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod2() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_class_method2.py",
        "MyClass.the_class_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method3.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method4.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method5.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/188.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#188 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method2.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method3.py", "raffi", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedMethod4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method4.py", "raffi", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method5.py", "raffi", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method6.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method7.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method8.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * This decorator isn't defined. Thus, we shouldn't have a CG node for it.
   *
   * <p>We now require nodes for functions under test. Otherwise, a test could pass even though the
   * function doesn't exist.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod9() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method9.py", "f", 0, 0);
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod10() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method10.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedMethod11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method11.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod12() throws ClassHierarchyException, CancelException, IOException {
    // TODO: Change to 1, 1, 2 once https://github.com/wala/ML/issues/188 is fixed.
    test("tf2_test_decorated_method12.py", "f", 0, 0);
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod13() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method13.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedFunctions()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_decorated_functions.py",
        "dummy_fun",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "dummy_test",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function2",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function3",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function4",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a pytest with decorators. */
  @Test
  public void testDecoratedFunctions2()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions.py", "test_dummy", 0, 0);
  }

  /**
   * Test a pytest without decorators that needs a PYTHONPATH. This is a "control" case. We'll add a
   * decorator in the next case.
   *
   * @see TestModules#testModule11().
   */
  @Test
  public void testDecoratedFunctions3()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj48/src/__init__.py",
          "proj48/src/tf2_test_module9a.py",
          "proj48/src/tf2_test_module9b.py",
          "proj48/src/test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj48",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test a pytest without decorators. This is a "control." */
  @Test
  public void testDecoratedFunctions4()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a pytest with a decorator. */
  @Test
  public void testDecoratedFunctions5()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions3.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test a pytest with a decorator that needs a PYTHONPATH.
   *
   * @see TestModules#testModule11().
   */
  @Test
  public void testDecoratedFunctions6()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj49/src/__init__.py",
          "proj49/src/tf2_test_module9a.py",
          "proj49/src/tf2_test_module9b.py",
          "proj49/src/test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj49",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test a Pytest with a decorator without parameters. */
  @Test
  public void testDecoratedFunctions7()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions4.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test a Pytest with a decorator without parameters that needs a PYTHONPATH.
   *
   * @see TestModules#testModule11().
   */
  @Test
  public void testDecoratedFunctions8()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj50/src/__init__.py",
          "proj50/src/tf2_test_module10a.py",
          "proj50/src/tf2_test_module10b.py",
          "proj50/src/test_module11.py"
        },
        "src/tf2_test_module10b.py",
        "D.f",
        "proj50",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test a Pytest with a decorator without parameters. The "test" is at the end of the filename.
   */
  @Test
  public void testDecoratedFunctions9()
      throws ClassHierarchyException, CancelException, IOException {
    test("decorated_function_test.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecorator()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator2.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator3.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testDecorator4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator4.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator5.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator6.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator7.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator8.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator9.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator10.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator11.py", "C.returned", 1, 1, Map.of(3, Set.of(TENSOR_5_INT32)));
  }

  /**
   * The {@code returned(a)} parameter is {@code a = tf.constant([1, 1.0])}, i.e. {@code (2,)
   * float32}. The asserted set is a union across contexts (per the test helper's union-per-vn
   * semantics): the {@code (2,) float32} is the parameter's real type, now precise after the top_k
   * output-shape composer (<a href="https://github.com/wala/ML/issues/609">wala/ML#609</a>)
   * sharpened the previous ⊤. The {@code (2,) int32} is a top_k {@code indices} output leaking in
   * through a collapsed 1-CFA context (it was already present in the old union as ⊤ int32); the
   * composer only made it concrete. The leak itself is a context-sensitivity artifact.
   */
  @Test
  public void testDecorator12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_decorator12.py",
        "returned",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_2_INT32)));
  }
}
