package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a {@code collections.namedtuple} instance's field reads give the values its
 * constructor was passed, whether by keyword or by position, including a field read off a loop
 * variable that carries the instance through {@code tf.while_loop}.
 */
public class TestNamedTupleKeyword extends AbstractTensorTest {

  private static final String FILE = "tf2_test_namedtuple_keyword.py";

  /**
   * A field of an instance built by keyword.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testKeyword() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * A field of an instance built by position.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPositional() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_positional", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * A field read off a loop variable holding an instance, reshaped to {@code (1, 1)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLoopField() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 1, 1))));
  }
}
