package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a keyword argument at a call to a summarized method binds the formal of that name (<a
 * href="https://github.com/wala/ML/issues/996">wala/ML#996</a>): a method summary is resolved
 * through the function class registered for it, where the bypass selector used to discard the
 * summary's parameter names, so the keyword bound nothing.
 */
public class TestKeywordSummaryMethod extends AbstractTensorTest {

  /** A pass-through layer called with its input by keyword returns the input's type. */
  @Test
  public void testKeywordToSummaryMethod()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_keyword_summary_method.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }
}
