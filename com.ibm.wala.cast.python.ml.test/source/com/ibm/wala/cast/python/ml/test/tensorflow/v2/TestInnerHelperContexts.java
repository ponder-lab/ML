package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a helper called inside another helper, which is called once with features and once with an
 * index (wala/ML#1305). The outer helper passes its parameter on, so the inner one has a node for
 * each call of the outer one, and what it computes for the features stays apart from what it
 * computes for the index.
 */
public class TestInnerHelperContexts extends AbstractTensorTest {

  private static final String FILE = "tf2_test_inner_helper_contexts.py";

  /** The index as it runs. */
  private static final TensorType INDEX = TensorType.of(INT_32, 5);

  /** The features as they run. */
  private static final TensorType FEATURES = TensorType.of(FLOAT_32, 5, 4);

  /** The index converted through the helper between reads its own type alone. */
  @Test
  public void testIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /** The features converted through the helper between read their own type alone. */
  @Test
  public void testFeatures() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_features", 1, 1, Map.of(2, Set.of(FEATURES)));
  }

  /**
   * The control: the function called twice calls the converting helper directly, and the index
   * reads its own type alone.
   */
  @Test
  public void testDirectIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_direct_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /** {@link #testIndex} with a targeted depth of 3. */
  @Test
  public void testIndexDepth3() throws ClassHierarchyException, CancelException, IOException {
    test(3, FILE, "consume_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /** {@link #testIndex} with a targeted depth of 4. */
  @Test
  public void testIndexDepth4() throws ClassHierarchyException, CancelException, IOException {
    test(4, FILE, "consume_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /**
   * The issue's example: the index copied by {@code np.array} through the helper between reads its
   * own type alone.
   */
  @Test
  public void testCopiedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_copied_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /** {@link #testCopiedIndex} with a targeted depth of 3. */
  @Test
  public void testCopiedIndexDepth3() throws ClassHierarchyException, CancelException, IOException {
    test(3, FILE, "consume_copied_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /** {@link #testCopiedIndex} with a targeted depth of 4. */
  @Test
  public void testCopiedIndexDepth4() throws ClassHierarchyException, CancelException, IOException {
    test(4, FILE, "consume_copied_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /**
   * The index at a targeted depth of 0, which keeps the framework methods context-insensitive even
   * under a program caller with two sites: their one node per method reads both calls' shapes. With
   * a depth of at least 3 under such a caller, the index reads its own type alone.
   */
  @Test
  public void testIndexDepth0() throws ClassHierarchyException, CancelException, IOException {
    test(0, FILE, "consume_index", 1, 1, Map.of(2, Set.of(INDEX, TensorType.of(INT_32, 5, 4))));
  }

  /** {@link #testIndexDepth0} for the index copied by {@code np.array}. */
  @Test
  public void testCopiedIndexDepth0() throws ClassHierarchyException, CancelException, IOException {
    test(
        0,
        FILE,
        "consume_copied_index",
        1,
        1,
        Map.of(2, Set.of(INDEX, TensorType.of(INT_32, 5, 4))));
  }
}
