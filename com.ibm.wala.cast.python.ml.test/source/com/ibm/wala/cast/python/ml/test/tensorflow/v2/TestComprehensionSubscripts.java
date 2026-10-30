package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A subscript of a tensor or ndarray captured from the enclosing scope, inside a list
 * comprehension, is typed as the same subscript in the enclosing function is.
 */
public class TestComprehensionSubscripts extends AbstractTensorTest {

  private static final String FILE = "tf2_test_comprehension_subscript.py";

  /** A control: the row read in a {@code for} loop. */
  @Test
  public void testLoopRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** An ndarray's row read inside a comprehension, indexed by the comprehension variable. */
  @Test
  public void testComprehensionRow() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_comprehension", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A tensor's row read inside a comprehension, indexed by the comprehension variable. */
  @Test
  public void testComprehensionTensorRow()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_tensor_row", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /** A tensor's row read inside a comprehension at a constant index. */
  @Test
  public void testComprehensionConstantRow()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_constant_row", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * An int parameter of a function that is not a comprehension's is still not read as an index, so
   * {@code t[i]} in {@code pick(t, i)} stays unseeded: the function is reached and its parameter
   * holds no tensor.
   */
  @Test
  public void testParameterIndexOutsideComprehension()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_parameter_index", 0, 0, Map.of());
  }
}
