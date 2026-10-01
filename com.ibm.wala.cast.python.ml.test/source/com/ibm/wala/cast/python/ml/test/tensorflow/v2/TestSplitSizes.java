package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code tf.split} by a list of sizes gives pieces whose extent along the split axis is
 * one of the listed sizes (<a href="https://github.com/wala/ML/issues/993">wala/ML#993</a>), where
 * the extent read as unresolved. The pieces are read by iterating the result, so each sink sees
 * every piece.
 */
public class TestSplitSizes extends AbstractTensorTest {

  private static final String FILE = "tf2_test_split_sizes.py";

  /** Equal sizes: every piece has the one listed extent. */
  @Test
  public void testEqualSizes() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_equal", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** Unequal sizes: a piece has either listed extent. */
  @Test
  public void testUnequalSizes() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unequal",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 3), TensorType.of(FLOAT_32, 3, 3))));
  }

  /** An inferred size ({@code -1}) is what the other sizes leave of the axis. */
  @Test
  public void testInferredSize() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_inferred",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 1, 3), TensorType.of(FLOAT_32, 3, 3))));
  }

  /** A split along a non-leading axis replaces that axis's extent. */
  @Test
  public void testNonLeadingAxis() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_axis",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 1), TensorType.of(FLOAT_32, 4, 2))));
  }
}
