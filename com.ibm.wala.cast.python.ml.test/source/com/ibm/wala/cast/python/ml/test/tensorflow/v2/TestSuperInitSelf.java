package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a base constructor reached through {@code super().__init__(...)} writes its fields to
 * the instance under construction (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>). The methods a {@code super()}
 * object binds took their {@code self} from a value the super stub never passes, so the base
 * constructor ran with an empty {@code self} and every field it wrote was lost, while its other
 * arguments bound correctly.
 */
public class TestSuperInitSelf extends AbstractTensorTest {

  /** The field read back right after the write, inside the base constructor. */
  @Test
  public void testReadInBase() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_init_self.py",
        "consume_in_base",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  /** The field read in the subclass constructor after the super call. */
  @Test
  public void testReadAfterSuper() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_init_self.py",
        "consume_after_super",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  /** The field read in the instance's own {@code call}. */
  @Test
  public void testReadInCall() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_super_init_self.py",
        "consume_in_call",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }
}
