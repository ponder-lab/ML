package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
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

  /** Datasets chosen by a branch and then transformed. */
  private static final String BATCHED_FILE = "tf2_test_fit_structured_component_batched.py";

  /**
   * A {@code float32} batch of rows of the given width whose batch extent is the reshape-style
   * placeholder, as a zipped dataset's batch reads its remainder.
   */
  private static TensorType unbatched(int width) {
    return new TensorType(FLOAT_32, List.of(new SymbolicDim("?"), new NumericDim(width)));
  }

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

  /** A tuple component built by {@code map}'s function is no tensor. */
  @Test
  public void testMapTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_map_pair_inputs", 0, 0, Map.of());
  }

  /**
   * A member of the tuple component {@code map} builds: a tensor, of unknown type since the members
   * of a mapped element's component are not resolved here; that is unchanged by the structure
   * reading.
   */
  @Test
  public void testMapTupleMember() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_map_pair_b", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * A flat component {@code map} returns stays a tensor: the {@code false} arm of the mapped
   * structure test. Its type is the join of the receiver's element members, {@code (2, 3)} and
   * {@code (2)} beside the runtime {@code (2, 4)}, an imprecision of the mapped component reading
   * that the structure reading leaves as it is.
   */
  @Test
  public void testMapFlatInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_map_flat_inputs",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                TensorType.of(FLOAT_32, 2, 3),
                TensorType.of(FLOAT_32, 2))));
  }

  /** A tuple component {@code zip} builds from a tuple of datasets is no tensor. */
  @Test
  public void testZipTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zip_pair_inputs", 0, 0, Map.of());
  }

  /**
   * A member of the tuple component {@code zip} builds from a tuple of datasets: a tensor, typed as
   * the targets' rows, a misreading of the nested tuple of datasets that the structure reading
   * leaves as it is. The component itself is no tensor with or without the structure reading, since
   * the zip types no tensor at its first selector.
   */
  @Test
  public void testZipTupleMember() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_zip_pair_m",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2),
                new TensorType(FLOAT_32, List.of(new SymbolicDim("?"))))));
  }

  /**
   * A flat component of a zip of datasets stays a tensor, with the zip batch's remainder read as a
   * placeholder extent beside the full batch.
   */
  @Test
  public void testZipFlatInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_zip_flat_inputs",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4), unbatched(4))));
  }

  /**
   * A dataset chosen by a branch between two with tuple components, in one frame where the choice
   * is either, has a tuple component.
   */
  @Test
  public void testBranchTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_branch_pair_inputs", 0, 0, Map.of());
  }

  /**
   * A dataset chosen by a branch between a tuple component and a flat one, in one frame where the
   * choice is either, is either at run time: the flat dataset's tensor, {@code (2, 6)}, or the
   * tuple, which is no tensor. The read keeps the tensor and adds a wholly unknown member, so no
   * reader takes {@code (2, 6)} for every value the component has.
   */
  @Test
  public void testBranchMixedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_branch_mixed_inputs",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), new TensorType(UNKNOWN, null))));
  }

  /**
   * An iterator over a dataset with a tuple component has a tuple component: the iterator is read
   * through the dataset it iterates.
   */
  @Test
  public void testIteratorTupleInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_iter_pair_inputs", 0, 0, Map.of());
  }

  /** A tuple component a {@code from_generator} signature declares is no tensor. */
  @Test
  public void testSignatureTupleInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_signature_pair_inputs", 0, 0, Map.of());
  }

  /** A member of the tuple component a {@code from_generator} signature declares. */
  @Test
  public void testSignatureTupleMember()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_signature_pair_m", 1, 1, Map.of(2, Set.of(TENSOR_NONE_3_FLOAT32)));
  }

  /** A flat component a {@code from_generator} signature declares stays a tensor. */
  @Test
  public void testSignatureFlatInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_signature_flat_inputs", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /** A tuple component the legacy {@code output_types} declare is no tensor. */
  @Test
  public void testTypesTupleInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_types_pair_inputs", 0, 0, Map.of());
  }

  /** A flat component the legacy {@code output_types} declare stays a tensor. */
  @Test
  public void testTypesFlatInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_types_flat_inputs", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /** A zipped dataset's whole element that is a tuple is a tuple component, no tensor. */
  @Test
  public void testZipElementInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zip_element_inputs", 0, 0, Map.of());
  }

  /**
   * A member of the zipped dataset's tuple element: a tensor, with the zip batch's remainder extent
   * beside the full batch, as {@link #testZipFlatInputs()} reads it.
   */
  @Test
  public void testZipElementMember() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_zip_element_m",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), unbatched(3))));
  }

  /** A zipped dataset's element {@code ((x, m), z)} is a tuple component. */
  @Test
  public void testZipDeepInputs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zip_deep_inputs", 0, 0, Map.of());
  }

  /** The first member of that component, {@code (x, m)}, is a tuple too. */
  @Test
  public void testZipDeepPair() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_zip_deep_pair", 0, 0, Map.of());
  }

  /**
   * The second member of that component is a tensor, with the zip batch's remainder extent beside
   * the full batch.
   */
  @Test
  public void testZipDeepMember() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_zip_deep_z",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5), unbatched(5))));
  }

  /**
   * As {@link #testBranchMixedInputs()}, with a frame per choice: the frame choosing the flat
   * dataset has the flat dataset's tensor, {@code (2, 6)}, and the frame choosing the tuple has
   * none.
   */
  @Test
  public void testSplitMixedInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE, "consume_split_mixed_inputs", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6))));
  }

  /**
   * A {@code train_step} override's {@code x} unpacked from the step's data, where the dataset is
   * chosen by a branch between a tuple component and a flat one in one frame, is that component
   * read through the unpack: the flat dataset's tensor, {@code (2, 6)}, beside a wholly unknown
   * member, as {@link #testBranchMixedInputs()} reads it in {@code call}.
   */
  @Test
  public void testUnpackMixed() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unpack_mixed",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), new TensorType(UNKNOWN, null))));
  }

  /**
   * An elementwise product of the {@code x} of {@link #testUnpackMixed()} reads its operand's dtype
   * as a partial union, {@code float32} beside the {@code UNKNOWN} marker, since the component may
   * be a structure, and not as a certain {@code float32}. Its shape is the resolvable subset,
   * {@code (2, 6)}.
   */
  @Test
  public void testUnpackScaled() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unpack_scaled",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 6),
                new TensorType(UNKNOWN, List.of(new NumericDim(2), new NumericDim(6))))));
  }

  /**
   * As {@link #testBranchMixedInputs()}, with the batch applied after the branch, {@code (a if flag
   * else b).batch(2)}: the batch's receiver is either dataset, so the component is the flat
   * dataset's tensor, batched, {@code (2, 6)}, beside a wholly unknown member for the tuple, and
   * not the tensor of whichever receiver is read first.
   */
  @Test
  public void testBatchedMixedInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        BATCHED_FILE,
        "consume_mixed_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), new TensorType(UNKNOWN, null))));
  }

  /**
   * As {@link #testBranchTupleInputs()}, with the batch applied after the branch: a tuple component
   * on both datasets is no tensor.
   */
  @Test
  public void testBatchedTupleInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(BATCHED_FILE, "consume_pair_batched", 0, 0, Map.of());
  }

  /**
   * A flat component on both datasets of a branch, batched after it, is either dataset's tensor,
   * batched: {@code (2, 6)} and {@code (2, 7)}.
   */
  @Test
  public void testBatchedFlatInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        BATCHED_FILE,
        "consume_flat_batched",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), TensorType.of(FLOAT_32, 2, 7))));
  }

  /**
   * As {@link #testBatchedMixedInputs()}, under two transformations, {@code (a if flag else
   * b).shuffle(4).batch(2)}: the batch's receiver is one shuffle, whose receiver is either dataset.
   */
  @Test
  public void testShuffledMixedInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        BATCHED_FILE,
        "consume_mixed_shuffled",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), new TensorType(UNKNOWN, null))));
  }

  /**
   * A tuple component on both datasets of a branch, of different arities, under two
   * transformations, is no tensor: each receiver has a structure there.
   */
  @Test
  public void testShuffledPairOrTripleInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(BATCHED_FILE, "consume_pair_or_triple_shuffled", 0, 0, Map.of());
  }

  /**
   * As {@link #testShuffledMixedInputs()}, with the flat dataset as the branch's first choice: the
   * reading does not depend on which receiver comes first.
   */
  @Test
  public void testShuffledFlatOrPairInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        BATCHED_FILE,
        "consume_flat_or_pair_shuffled",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 6), new TensorType(UNKNOWN, null))));
  }

  /**
   * A {@code map} whose function takes the element's components as parameters, {@code lambda inp,
   * y: inp}, returns one component and not the element, so the mapped element is that component,
   * the tuple {@code (x, mask)}, and its inputs component is the tensor {@code x}, batched, {@code
   * (2, 4)}, not the structure the receiver's element has there.
   */
  @Test
  public void testMapComponentInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        BATCHED_FILE,
        "consume_map_component",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }
}
