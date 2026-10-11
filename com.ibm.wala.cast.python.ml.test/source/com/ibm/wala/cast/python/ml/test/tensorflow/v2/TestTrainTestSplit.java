package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests scikit-learn's {@code train_test_split}. Each half of a split is a new list whose elements
 * are some of the input's, so a list of arrays split this way keeps its elements' types. Unmodeled,
 * the halves had no elements the analysis could read, and every function a half reached was
 * untyped.
 */
public class TestTrainTestSplit extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_train_test_split.py";

  /** An element of the train half of a list of {@code (2, 3)} float32 arrays. */
  @Test
  public void testTrainElement() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_element", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** An element of the test half. */
  @Test
  public void testTestElement() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_test_element", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * The train half of an array. Its row count is a fraction of the input's, so it must not carry
   * the input's {@code (10, 4)}. The halves are modeled as lists, so it is reached with no tensor
   * type rather than with a wrong one.
   */
  @Test
  public void testArrayHalf() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_array", 0, 0);
  }
}
