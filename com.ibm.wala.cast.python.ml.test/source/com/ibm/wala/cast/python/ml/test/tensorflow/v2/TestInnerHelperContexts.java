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
 * index (wala/ML#1305). The inner helper has one node for every call of the outer one, so what it
 * computes for the features joins what it computes for the index.
 */
public class TestInnerHelperContexts extends AbstractTensorTest {

  private static final String FILE = "tf2_test_inner_helper_contexts.py";

  /** The index as it runs. */
  private static final TensorType INDEX = TensorType.of(INT_32, 5);

  /** The features as they run. */
  private static final TensorType FEATURES = TensorType.of(FLOAT_32, 5, 4);

  /**
   * What a value converted by {@code convert} reads: each call's shape with each call's dtype, and
   * with an unknown dtype besides.
   */
  private static final Set<TensorType> CONVERTED_JOIN =
      Set.of(
          INDEX,
          TensorType.of(FLOAT_32, 5),
          TensorType.of(UNKNOWN, 5),
          FEATURES,
          TensorType.of(INT_32, 5, 4),
          TensorType.of(UNKNOWN, 5, 4));

  /**
   * What a value copied by {@code copy_through} reads: each call's shape with each call's dtype.
   */
  private static final Set<TensorType> COPIED_JOIN =
      Set.of(INDEX, TensorType.of(FLOAT_32, 5), FEATURES, TensorType.of(INT_32, 5, 4));

  /**
   * The index converted through the helper between: it reads the features' types beside its own.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_index", 1, 1, Map.of(2, CONVERTED_JOIN));
  }

  /**
   * The features converted through the helper between: they read the index's types beside their
   * own.
   *
   * <p>TODO: Reads {@link #FEATURES} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testFeatures() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_features", 1, 1, Map.of(2, CONVERTED_JOIN));
  }

  /**
   * The control: the function called twice calls the converting helper directly, and the index
   * reads its own type alone.
   */
  @Test
  public void testDirectIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_direct_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /**
   * {@link #testIndex} with a targeted depth of 3, which does not lengthen the string of a program
   * function's callee.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testIndexDepth3() throws ClassHierarchyException, CancelException, IOException {
    test(3, FILE, "consume_index", 1, 1, Map.of(2, CONVERTED_JOIN));
  }

  /**
   * {@link #testIndex} with a targeted depth of 4.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testIndexDepth4() throws ClassHierarchyException, CancelException, IOException {
    test(4, FILE, "consume_index", 1, 1, Map.of(2, CONVERTED_JOIN));
  }

  /**
   * The issue's example: the index copied by {@code np.array} through the helper between.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testCopiedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_copied_index", 1, 1, Map.of(2, COPIED_JOIN));
  }

  /**
   * {@link #testCopiedIndex} with a targeted depth of 3.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testCopiedIndexDepth3() throws ClassHierarchyException, CancelException, IOException {
    test(3, FILE, "consume_copied_index", 1, 1, Map.of(2, COPIED_JOIN));
  }

  /**
   * {@link #testCopiedIndex} with a targeted depth of 4.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 is fixed.
   */
  @Test
  public void testCopiedIndexDepth4() throws ClassHierarchyException, CancelException, IOException {
    test(4, FILE, "consume_copied_index", 1, 1, Map.of(2, COPIED_JOIN));
  }
}
