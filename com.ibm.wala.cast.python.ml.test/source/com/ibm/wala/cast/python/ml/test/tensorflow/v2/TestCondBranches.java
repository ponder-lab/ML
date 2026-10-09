package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code tf.cond} calls both of its branches, so a function reached only through {@code
 * false_fn} gets a call-graph node and its parameters are typed (<a
 * href="https://github.com/wala/ML/issues/1029">wala/ML#1029</a>).
 */
public class TestCondBranches extends AbstractTensorTest {

  private static final String FILE = "tf2_test_cond_false_branch.py";

  /**
   * The false branch's callee reads the int32 value the branch passes it.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testFalseBranchCallee() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_false", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * The true branch's callee keeps its own float32 value, and nothing of the false branch's.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTrueBranchCallee() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_true", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }
}
