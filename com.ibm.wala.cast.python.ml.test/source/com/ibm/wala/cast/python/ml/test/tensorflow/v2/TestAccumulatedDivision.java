package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a tensor accumulated over a list of layers in a loop and then divided (<a
 * href="https://github.com/wala/ML/issues/1094">wala/ML#1094</a>). The loop variable and its {@code
 * +=} read each other, so the first dtype read of either is made while the other is still
 * unresolved; that read must not decide the dtype. Each spelling feeds its own consumer, whose
 * parameter is read in its own {@code call}, with no sink.
 */
public class TestAccumulatedDivision extends AbstractTensorTest {

  private static final String FILE = "tf2_test_accumulate_divide.py";

  private static final TensorType TENSOR_2_5_8_FLOAT32 = TensorType.of(FLOAT_32, 2, 5, 8);

  /**
   * Accumulated in a loop and divided by the list's length: the dtype is the layers' {@code
   * float32}. The shape the {@code len(...)} divisor loses is {@link #testLoopLenShape()}.
   */
  @Test
  public void testLoopLen() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "ConsumeLoopLen.call", 1, 1, Map.of(3, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * As {@link #testLoopLen()}, with the shape the runtime asserts.
   *
   * <p>TODO: Pass once https://github.com/wala/ML/issues/1097 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testLoopLenShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "ConsumeLoopLen.call", 1, 1, Map.of(3, Set.of(TENSOR_2_5_8_FLOAT32)));
  }

  /** Accumulated in a loop and divided by a literal: the type is exact. */
  @Test
  public void testLoopTwo() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "ConsumeLoopTwo.call", 1, 1, Map.of(3, Set.of(TENSOR_2_5_8_FLOAT32)));
  }

  /** Accumulated without a loop and divided by the list's length: the dtype is {@code float32}. */
  @Test
  public void testNoLoopLen() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "ConsumeNoLoopLen.call", 1, 1, Map.of(3, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * As {@link #testNoLoopLen()}, with the shape the runtime asserts.
   *
   * <p>TODO: Pass once https://github.com/wala/ML/issues/1097 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testNoLoopLenShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "ConsumeNoLoopLen.call", 1, 1, Map.of(3, Set.of(TENSOR_2_5_8_FLOAT32)));
  }
}
