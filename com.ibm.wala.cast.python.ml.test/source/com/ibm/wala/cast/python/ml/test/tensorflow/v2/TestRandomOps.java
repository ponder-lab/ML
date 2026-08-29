package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_30_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_5_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of random-distribution allocator inference ({@code tf.random.gamma}/{@code poisson}/ {@code
 * truncated_normal}), regrouped per the wala/ML#635 Phase 3 rulings; the assertions are verbatim.
 */
public class TestRandomOps extends AbstractTensorTest {

  /**
   * Regression guard for wala/ML#449's closing fix: {@code tf.random.truncated_normal(shape)} now
   * dispatches to the {@code TruncatedNormal} generator (via {@code PROPERTY_NAME_GENERATORS}) and
   * resolves to precise {@code (2, 3) float32}. Pre-fix this fell through to {@code
   * ReadDataFallback} and emitted {@code ⊤ shape / UNKNOWN dtype}.
   */
  @Test
  public void testTruncatedNormal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_truncated_normal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testGamma()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testGamma2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testGamma3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testGamma4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGammaMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testPoisson()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testPoisson4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testGamma7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testPoisson5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson5.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.gamma} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when the {@code shape} argument
   * is unresolvable (here from {@code json.loads}) the output rank can't be known, so the result
   * floors to ⊤ rather than throwing (which previously aborted the whole analysis). The dtype stays
   * float32.
   */
  @Test
  public void testGammaUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.poisson} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): same as {@link
   * #testGammaUnresolvable()}, the output rank rides on the unresolvable {@code shape}, so the
   * result floors to ⊤ rather than throwing.
   */
  @Test
  public void testPoissonUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * The reported witness of <a href="https://github.com/wala/ML/issues/827">wala/ML#827</a>: {@code
   * tf.Variable(rng.randn())} over the module-alias form. A shapeless draw is a Python number, so
   * the variable takes TensorFlow's {@code float32} weak default rather than NumPy's {@code
   * float64}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNpRandomScalarDraw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_random.py", "consume", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * The array half of {@link #testNpRandomScalarDraw()}: {@code np.random.randn(2, 3)} names its
   * shape variadically and draws {@code float64}, so the same call site's arity decides both axes.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNpRandomRandn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_random.py", "consume2", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT64)));
  }

  /**
   * The rank-1 variadic draw, {@code np.random.rand(4)}: the sibling distribution of {@link
   * #testNpRandomRandn()} under the same signature shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNpRandomRand()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_random.py", "consume3", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT64)));
  }

  /**
   * The sized draw, {@code np.random.normal(0.0, 1.0, (5, 2))}: the shape is the {@code size}
   * argument, so the two distribution parameters ahead of it must not be read as dimensions.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNpRandomNormal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_random.py", "consume4", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT64)));
  }

  /**
   * The sibling distribution under the sized signature, {@code np.random.uniform(0.0, 1.0, (2,
   * 2))}: the two share a generator, so this pins that both names reach it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNpRandomUniform()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_random.py", "consume5", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT64)));
  }

  /**
   * The sized family's own shapeless draw, {@code np.random.normal()}: omitting {@code size} is the
   * scalar case that {@link #testNpRandomScalarDraw()} reaches through an empty argument list, so
   * the two recognitions are pinned separately.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNpRandomSizedScalarDraw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_random.py", "consume6", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * A seeded generator object reaches the same draws as the module-level surface, so `size` shapes
   * the result exactly as it does for `np.random.uniform` (wala/ML#827).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRandomStateUniform()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_random_state.py",
        "consume_uniform",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 2, 20))));
  }

  /**
   * The same generator's normal draw (wala/ML#827).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRandomStateNormal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_random_state.py",
        "consume_normal",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 3, 4))));
  }

  /**
   * The draw-then-narrow idiom: the cast keeps the drawn shape and changes only the dtype
   * (wala/ML#827). Before the generator was modeled this reached a consumer with its dtype
   * recovered and its shape absent, which is the dangerous half of that pair.
   *
   * <p>Regression guard for <a href="https://github.com/wala/ML/issues/849">wala/ML#849</a>, where
   * converting the narrowed array unioned a ⊤-shaped member beside the exact one and summarized the
   * whole value to unknown rank. The conversion reached the narrowing's allocation and found no
   * generator anchored on it; the receiver is never passed as an argument, so it now resolves
   * through the caller-aware receiver walk.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRandomStateDrawThenCast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_random_state.py",
        "consume_cast",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 20))));
  }

  /**
   * Control for {@link #testRandomStateDrawThenCast()}: the module-level surface with the same
   * narrowing.
   *
   * <p>This is the control that established the extra member of <a
   * href="https://github.com/wala/ML/issues/849">wala/ML#849</a> did not belong to the generator
   * object, since both surfaces produced it identically. It keeps that role now that the defect is
   * fixed: the two surfaces must stay in agreement, so a regression reaching only one of them fails
   * here rather than passing unnoticed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModuleDrawThenCastControl()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_random_state.py",
        "consume_module_cast",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 20))));
  }

  /**
   * Isolates the tensor conversion from the narrowing (wala/ML#849): consuming a narrowed array
   * directly, without converting it, is exact and single-membered. Together with the two blocked
   * cases above, this places the extra member on the conversion of a narrowed value rather than on
   * the narrowing, which an earlier reading of this issue had it on.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRawCastProbe()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_random_state.py",
        "consume_raw_cast",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 20))));
  }
}
