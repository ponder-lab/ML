package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The inputs a model's {@code call} gets from {@code fit} over a map of a {@code TFRecordDataset}
 * whose function returns {@code (x, y)}: {@code x} alone, so {@code y}'s shape does not reach them.
 */
public class TestFitTFRecordMap extends AbstractTensorTest {

  private static final String FILE = "tf2_test_fit_tfrecord_map.py";

  /**
   * The inputs are the batched {@code x}, full or partial, without {@code y}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testInputs() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_fit_inputs",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), new NumericDim(4))))));
  }

  /**
   * A map whose function returns {@code (x, y)} on one call and {@code x} alone on another, which
   * is the dataset {@code fit} is given: the inputs are the batched {@code x}, without the {@code
   * y} of the call whose dataset {@code fit} is not given.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMaybeLabelledInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_map_maybe_labelled.py",
        "consume_fit_inputs",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * A map whose function returns {@code (x, y)} on one call and a tensor of another shape on
   * another, which is the dataset {@code fit} is given: the inputs are that tensor batched, which
   * reading the element where it is a tensor, rather than as the pair's members, gives, and not the
   * pair's label. The batched {@code x} is imprecision: the analysis cannot tell which call returns
   * the pair, so the step may take the pair's first member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMaybeUnlabelledOtherInputs()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_map_maybe_unlabelled_other.py",
        "consume_fit_inputs",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * The first input of a model fed a zip whose first dataset is a map over a {@code
   * TFRecordDataset} returning a pair: {@code call} unpacks its inputs {@code (a, b)}, and {@code
   * a} is the batched first member of the pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNestedFirstInput() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_fit_uncopied_nested.py",
        "consume_first_input",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 4),
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), new NumericDim(4))))));
  }
}
