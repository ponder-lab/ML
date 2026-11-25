package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A call of {@code update}, {@code setdefault} or {@code add} in a function the call graph does not
 * reach changes no list, since no list has those methods, so a slice of a list literal keeps the
 * literal's positions (<a href="https://github.com/wala/ML/issues/1328">wala/ML#1328</a>).
 */
public class TestUnreachedDictMethods extends AbstractTensorTest {

  /**
   * A layer's slice {@code inputs[0:-1]} holds the data inputs alone, though its unreached {@code
   * get_config} updates what {@code super().get_config()} returns.
   */
  @Test
  public void testSliceKeepsPositions()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_unreached_dict_methods.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
