package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests helpers called under a method, each called once with features and once with an index
 * (wala/ML#1305). A call under a method is keyed on the method's node rather than on a call string,
 * so a helper's callee had one node for every node of the helper under one object; a helper that
 * passes its parameter on now keys its call on its own node.
 */
public class TestMethodHelperContexts extends AbstractTensorTest {

  private static final String FILE = "tf2_test_method_helper_contexts.py";

  /** The index as it runs. */
  private static final TensorType INDEX = TensorType.of(INT_32, 5);

  /** The features as they run. */
  private static final TensorType FEATURES = TensorType.of(FLOAT_32, 5, 4);

  /** The index converted by a function nested in a method reads its own type alone. */
  @Test
  public void testNestedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_nested_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /** The features converted by a function nested in a method read their own type alone. */
  @Test
  public void testNestedFeatures() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_nested_features", 1, 1, Map.of(2, Set.of(FEATURES)));
  }

  /** The index converted by a module function that a method calls reads its own type alone. */
  @Test
  public void testHelperIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_helper_index", 1, 1, Map.of(2, Set.of(INDEX)));
  }

  /**
   * The index converted by a recursive helper that a method calls: the helper's recursive calls
   * have one node for both of the method's calls, so the index reads the features' types beside its
   * own. A call a function makes to itself keeps one site, since a second would unroll the
   * recursion by one call whose nodes merge again below it.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 separates a recursive helper's callers.
   */
  @Test
  public void testWalkedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_walked_index",
        1,
        1,
        Map.of(
            2, Set.of(INDEX, TensorType.of(FLOAT_32, 5), FEATURES, TensorType.of(INT_32, 5, 4))));
  }

  /**
   * The index converted by two helpers that call each other, the first called by a method: their
   * recursive calls have one node for both of the method's calls, so the index reads the features'
   * types beside its own, as for {@link #testWalkedIndex}.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1305 separates a recursive helper's callers.
   */
  @Test
  public void testTickedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_ticked_index",
        1,
        1,
        Map.of(
            2, Set.of(INDEX, TensorType.of(FLOAT_32, 5), FEATURES, TensorType.of(INT_32, 5, 4))));
  }
}
