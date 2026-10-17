package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a loop over Python's builtin {@code reversed} (<a
 * href="https://github.com/wala/ML/issues/1099">wala/ML#1099</a>): {@code reversed} iterates its
 * argument, so the loop variable holds the argument's elements. Each loop passes the element to its
 * own consumer, whose parameter is read in its own body, with no sink.
 */
public class TestReversed extends AbstractTensorTest {

  private static final String FILE = "tf2_test_reversed.py";

  /** Over a rank-1 tensor, the loop variable is a scalar of the tensor's dtype. */
  @Test
  public void testReversedTensor() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale_reversed", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * The {@code reversed} object itself is no tensor: only a loop over it reads its argument's
   * elements, so a function handed {@code reversed(y)} is reached with no tensor parameter.
   */
  @Test
  public void testReversedObject() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "take_reversed", 0, 0);
  }

  /**
   * In the loop's own function, the {@code reversed(y)} object is not one of its tensors: the loop
   * reads {@code y}'s elements through it, but the object, a list iterator at run time, is no
   * tensor. Typing the call's result as {@code y} instead would add it to the tensors counted here.
   */
  @Test
  public void testReversedLoopLocals()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "reversed_param", 1, 6, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Over a list of tensors, the loop variable is one of the list's tensors. */
  @Test
  public void testReversedList() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale_reversed_list", 1, 2, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }
}
