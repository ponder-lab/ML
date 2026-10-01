package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a list of tensors passed to a Keras layer reaches the layer's {@code call} with its
 * elements typed, link by link along a group-dispatching kernel's chain (<a
 * href="https://github.com/wala/ML/issues/993">wala/ML#993</a>). Each variant ends in its own sink,
 * whose parameter is the first list element as the innermost layer's {@code call} reads it.
 */
public class TestLayerListInput extends AbstractTensorTest {

  private static final String FILE = "tf2_test_layer_list_input.py";

  private static final TensorType TENSOR_4_3_FLOAT32 = TensorType.of(FLOAT_32, 4, 3);

  private static final TensorType TENSOR_2_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 3);

  /** The list goes straight to the layer. */
  @Test
  public void testDirect() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_direct", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** An outer layer's {@code call} passes its list input on to an inner layer. */
  @Test
  public void testNested() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_nested", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** The outer layer passes a slice of its list input, dropping the trailing group tensor. */
  @Test
  public void testSliced() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sliced", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** The outer layer calls a layer it holds in a list attribute. */
  @Test
  public void testListed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_listed", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** The outer layer calls each layer it holds in a list attribute, by a loop index. */
  @Test
  public void testLooped() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_looped", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /** As {@link #testLooped()}, with the loop bound read from the list's length. */
  @Test
  public void testCounted() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_counted", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * Control: an index that is a parameter fed a literal is not a loop variable, so the read stays
   * exact rather than taking every element (the group tensor beside the one read).
   */
  @Test
  public void testConstantBoundIndex()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_bound", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * The outer layer builds the inner layer's list input by splitting and appending, iterating the
   * slice {@code inputs[0:-1]} that drops the group tensor.
   */
  @Test
  public void testBuilt() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_built", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The gate splits each input per subnet and hands each subnet, by a loop index, a list it builds
   * by appending.
   */
  @Test
  public void testDispatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_dispatched", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
