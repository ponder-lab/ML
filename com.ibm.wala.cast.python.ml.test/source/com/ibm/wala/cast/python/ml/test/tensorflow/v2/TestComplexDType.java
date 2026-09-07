package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Regression pin for <a href="https://github.com/wala/ML/issues/816">wala/ML#816</a>: an array
 * allocated with a complex {@code dtype} argument resolves to that complex dtype rather than
 * degrading to numpy's {@code float64} default. The {@code COMPLEX_64}/{@code COMPLEX_128} resolver
 * entries already existed; only the numpy-module fields were missing, so a complex dtype token had
 * no allocation to match and the argument was disregarded.
 *
 * <p>Asserting {@code complex64}/{@code complex128} positively is what makes this catch the
 * original defect: were the argument disregarded and the allocator to fall back to {@code float64}
 * again, these assertions would fail rather than pass.
 */
public class TestComplexDType extends AbstractTensorTest {

  @Test
  public void testComplexZerosResolvesToComplex64()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_complex_dtype.py",
        "consume_complex64",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_5_COMPLEX64)));
  }

  @Test
  public void testComplexOnesResolvesToComplex128()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_complex_dtype.py",
        "consume_complex128",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(COMPLEX_128, 2, 3))));
  }
}
