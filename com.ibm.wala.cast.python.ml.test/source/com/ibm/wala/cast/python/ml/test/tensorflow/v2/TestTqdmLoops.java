package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a loop over a dataset wrapped in {@code tqdm}, which yields the elements of the
 * iterable it wraps, types each element as the dataset's. Before, {@code tqdm} was not modeled, so
 * the element, and every key read on it, was a tensor of unknown type.
 */
public class TestTqdmLoops extends AbstractTensorTest {

  private static final String FILE = "tf2_test_tqdm_loop.py";

  /**
   * A batch of rows of three {@code float32} values. The four rows batch into two full batches, so
   * there is no partial batch.
   */
  private static final Set<TensorType> AUDIOS = Set.of(TensorType.of(FLOAT_32, 2, 3));

  /** The wrapped dataset enumerated, as an evaluation loop does. */
  @Test
  public void testEnumeratedTqdm() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "enumerated_sink", 1, 1, Map.of(2, AUDIOS));
  }

  /** The wrapped dataset iterated directly. */
  @Test
  public void testPlainTqdm() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "plain_sink", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /** A dataset an attribute holds, wrapped and enumerated in a method, as a trainer's does. */
  @Test
  public void testAttributeTqdm() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "attribute_sink", 1, 1, Map.of(2, AUDIOS));
  }
}
