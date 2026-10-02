package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a dictionary read by {@code get} or {@code pop} with a constant key whose default is a
 * variable (<a href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): the default flows to
 * the result whether it is a literal or a value the caller passed.
 */
public class TestDictDefault extends AbstractTensorTest {

  /** The default of a {@code get} with a constant key, passed in by the caller. */
  @Test
  public void testGetVariableDefault()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dict_default.py",
        "consume_get",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The default of a {@code pop} with a constant key, passed in by the caller. */
  @Test
  public void testPopVariableDefault()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dict_default.py",
        "consume_pop",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 5))));
  }
}
