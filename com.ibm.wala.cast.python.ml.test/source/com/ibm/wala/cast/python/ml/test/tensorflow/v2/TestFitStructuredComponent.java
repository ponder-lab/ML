package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the inputs a Keras default {@code train_step} hands the model's {@code call}, when the
 * dataset element's inputs component is itself a structure, are not typed as a tensor: {@code ((x,
 * mask), y)} hands {@code call} a tuple of two tensors and {@code ({"ids": ids, "m": m}, y)} a dict
 * of two, not one tensor of several shapes, while the members {@code call} reads keep their types.
 * Flat inputs, {@code (x, y)}, stay a tensor.
 */
public class TestFitStructuredComponent extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_structured_component.py";

  /** A tuple of inputs is no tensor. */
  @Test
  public void testTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_inputs", 0, 0, Map.of());
  }

  /** The first member of the tuple, unpacked in {@code call}. */
  @Test
  public void testTupleMemberX() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /** The second member of the tuple, unpacked in {@code call}. */
  @Test
  public void testTupleMemberMask() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_mask", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The first member of the tuple, subscripted in {@code call}. */
  @Test
  public void testTupleSubscript() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair_first", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A tuple of inputs of mixed dtypes is no tensor either, and in particular not the cross product
   * of its members' shapes and dtypes.
   */
  @Test
  public void testMixedTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_inputs", 0, 0, Map.of());
  }

  /** The {@code int32} member of the mixed tuple keeps its own shape and dtype alone. */
  @Test
  public void testMixedTupleMember() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mixed_ids", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /** A dict of inputs is no tensor. */
  @Test
  public void testDictInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_inputs", 0, 0, Map.of());
  }

  /** The {@code "ids"} entry of the dict, read by key in {@code call}. */
  @Test
  public void testDictEntryIds() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_ids", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 2, 3))));
  }

  /** The {@code "m"} entry of the dict, read by key in {@code call}. */
  @Test
  public void testDictEntryM() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyed_m", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5))));
  }

  /** Flat inputs are the tensor they are. */
  @Test
  public void testFlatInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_flat_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The tuple is no tensor parameter of {@code call} either; its locals {@code inputs[0]}, {@code
   * x} and {@code mask} are tensors.
   */
  @Test
  public void testTupleCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Pair.call", 0, 3, Map.of());
  }

  /** As {@link #testTupleCall()}, for the mixed tuple: its locals {@code x} and {@code ids}. */
  @Test
  public void testMixedTupleCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Mixed.call", 0, 2, Map.of());
  }

  /**
   * As {@link #testTupleCall()}, for the dict: its locals are the three reads by key, {@code
   * inputs["ids"]} and twice {@code inputs["m"]}.
   */
  @Test
  public void testDictCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Keyed.call", 0, 3, Map.of());
  }

  /** Flat inputs are a tensor parameter of {@code call}. */
  @Test
  public void testFlatCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Flat.call", 1, 1, Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
