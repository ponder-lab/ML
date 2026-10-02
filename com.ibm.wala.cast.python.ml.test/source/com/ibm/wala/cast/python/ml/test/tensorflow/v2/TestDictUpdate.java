package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a dictionary updated from another dictionary the caller passed in (<a
 * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): the argument's fields arrive
 * through its points-to set rather than from a literal's constants, and the receiver gains them.
 */
public class TestDictUpdate extends AbstractTensorTest {

  /** The field of the updated dictionary read by the key the argument wrote. */
  @Test
  public void testUpdateFromArgument()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dict_update_argument.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 4))));
  }
}
