package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a loop nested in another loop's body iterates through temporaries of its own, so the
 * outer loop's index reads only the outer iterable's elements (wala/ML#1106). Before, the two loops
 * could lower to temporaries of the same name in the same scope, and the inner loop's iterator
 * reached the outer loop's index through the outer loop's header.
 */
public class TestNestedLoopTemporaries extends AbstractTensorTest {

  private static final String FILE = "tf2_test_nested_loop_temporaries.py";

  /**
   * The index of an outer loop over {@code range}, in a module's body, is no tensor, though the
   * loop in its body iterates over tensors.
   */
  @Test
  public void testOuterIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "load", 0, 0);
  }

  /** The control: the inner loop's elements are the tensors it iterates over. */
  @Test
  public void testInnerElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_3_FLOAT32)));
  }

  /**
   * The index of an outer loop over {@code range}, in a function's body, is no tensor, though the
   * loop in its body iterates over a generator of tensors.
   */
  @Test
  public void testOuterIndexOverGenerator()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "load_gen", 0, 0);
  }

  /** The control: the inner loop's elements are the tensors the generator yields. */
  @Test
  public void testInnerElementOfGenerator()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_gen", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }
}
