package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A shape argument whose value contains itself (wala/ML#990), on {@code
 * tf2_test_shape_argument_cycle.py}: {@code shape = [shape, 2]} in a loop is one abstract list
 * whose first element is the list itself, and reading its nested elements recursed until the stack
 * was exhausted. The containers on the walk are now carried, and one met again reads as an
 * unresolvable nested form. The witness is completion with the result read as a {@code float32} of
 * unknown shape: on the previous engine this test ends in a {@code StackOverflowError}.
 */
public class TestShapeArgumentCycle extends AbstractTensorTest {

  @Test
  public void testSelfContainingShapeCompletes()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_shape_argument_cycle.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }
}
