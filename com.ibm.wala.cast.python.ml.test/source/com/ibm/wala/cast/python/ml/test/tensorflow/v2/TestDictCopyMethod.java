package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a dictionary's {@code copy()} returns a new dictionary holding the same entries
 * (wala/ML#1082). Before, the method was not modeled, so the copy pointed to nothing and the values
 * it held were lost to whatever received it.
 */
public class TestDictCopyMethod extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dict_copy_method.py";

  /** A value read off a dictionary's copy is the original's value. */
  @Test
  public void testCopyHoldsValues() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_features", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * A write to the copy does not reach the original, since the copy is a new dictionary rather than
   * the original itself.
   */
  @Test
  public void testCopyIsNewDictionary()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_original", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }
}
