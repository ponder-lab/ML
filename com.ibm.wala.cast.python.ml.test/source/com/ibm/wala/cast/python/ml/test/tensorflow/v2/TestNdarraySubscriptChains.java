package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Stage-by-stage probes over the numpy data-preparation chain the corpus's custom training loops
 * share ({@code np.ones} or a loaded array, a dtype-changing division, an {@code astype}, and the
 * dim-adding ellipsis/newaxis subscript), each pinned against the fixture's own runtime assertions.
 * The subscript-over-elementwise stages are the wala/ML#396 regression guards: a binary-op result
 * carries dataflow state but no allocation, and both the seeding walk and the creator walk
 * previously dropped it, so the subscript result was not a tensor at all.
 */
public class TestNdarraySubscriptChains extends AbstractTensorTest {

  @Test
  public void testRawArray()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_subscript_chain.py",
        "consume_raw",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UINT_8, 96, 28, 28))));
  }

  @Test
  public void testDivision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_subscript_chain.py",
        "consume_div",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 96, 28, 28))));
  }

  @Test
  public void testNewaxisOverDivision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_subscript_chain.py",
        "consume_axis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 96, 28, 28, 1))));
  }

  @Test
  public void testAstypeDivision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_subscript_chain.py",
        "consume_astype_div",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 96, 28, 28))));
  }

  @Test
  public void testDestructuredDivision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_subscript_chain.py",
        "consume_destructured",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 96, 28, 28))));
  }

  @Test
  public void testNewaxisOverAstypeDivision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ndarray_subscript_chain.py",
        "consume_astype_axis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 96, 28, 28, 1))));
  }
}
