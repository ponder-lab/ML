package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_30_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_5_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;

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
}
