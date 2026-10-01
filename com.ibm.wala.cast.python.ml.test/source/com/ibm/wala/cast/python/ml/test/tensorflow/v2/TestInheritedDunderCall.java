package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that calling a Keras layer whose {@code __call__} is inherited from a user base class
 * dispatches to that {@code __call__} (<a
 * href="https://github.com/wala/ML/issues/994">wala/ML#994</a>), where the dispatch went straight
 * to the layer's own {@code call}.
 */
public class TestInheritedDunderCall extends AbstractTensorTest {

  private static final String FILE = "tf2_test_inherited_dunder_call.py";

  private static final TensorType INPUT = TensorType.of(FLOAT_32, 2, 3, 8);

  /** The inherited {@code __call__} receives the call's input. */
  @Test
  public void testInheritedDunderCall()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inherited", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /** The layer's {@code call} is still reached, through the inherited {@code __call__}. */
  @Test
  public void testCallStillReached() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_inherited_call", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /** Control: a {@code __call__} defined on the layer's own class. */
  @Test
  public void testDirectDunderCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_direct", 1, 1, Map.of(2, Set.of(INPUT)));
  }

  /**
   * Control: a {@code __call__} on the layer's own class still reaches the layer's {@code call}.
   */
  @Test
  public void testDirectCallReached() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_direct_call", 1, 1, Map.of(2, Set.of(INPUT)));
  }
}
