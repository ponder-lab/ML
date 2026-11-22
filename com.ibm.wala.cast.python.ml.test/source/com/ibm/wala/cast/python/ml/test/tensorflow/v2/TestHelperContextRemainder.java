package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests helpers whose callers the call strings still join (wala/ML#1318): an argument computed from
 * a parameter rather than being one, which keeps one site, and a chain of helpers three deep, whose
 * innermost callee's two sites are the calls within the chain.
 */
public class TestHelperContextRemainder extends AbstractTensorTest {

  private static final String FILE = "tf2_test_helper_context_remainder.py";

  /** The index as it runs. */
  private static final TensorType INDEX = TensorType.of(INT_32, 5);

  /** The features as they run. */
  private static final TensorType FEATURES = TensorType.of(FLOAT_32, 5, 4);

  /**
   * What the index reads through either shape: each call's shape with each call's dtype, and with
   * an unknown dtype besides.
   */
  private static final Set<TensorType> JOINED =
      Set.of(
          INDEX,
          TensorType.of(FLOAT_32, 5),
          TensorType.of(UNKNOWN, 5),
          FEATURES,
          TensorType.of(INT_32, 5, 4),
          TensorType.of(UNKNOWN, 5, 4));

  /**
   * The index converted through a helper that passes a slice of its parameter on: the converter has
   * one node for both of the outer function's calls.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1318 is fixed.
   */
  @Test
  public void testComputedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_computed_index", 1, 1, Map.of(2, JOINED));
  }

  /**
   * The index converted through a chain of three helpers: the converter has one node for both of
   * the outer function's calls.
   *
   * <p>TODO: Reads {@link #INDEX} alone once wala/ML#1318 is fixed.
   */
  @Test
  public void testChainedIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_chained_index", 1, 1, Map.of(2, JOINED));
  }
}
