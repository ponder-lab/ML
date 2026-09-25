package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
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
 * A subscript of a value whose type arrives only through the dataflow, on {@code
 * musictx_sidecar_proj}: a data loader in the form of a music transformer's, whose {@code
 * Data._get_seq} unpickles a sequence (typed by the project's {@code ariadne-types.json} sidecar as
 * int64 {@code (n,)}), whose {@code batch} builds {@code np.array} over a comprehension of those
 * results, and whose {@code slide_seq2seq_batch} slices the batch as {@code data[:, :-1]}. The
 * sidecar's dtype reaches the batch with its shape, and the slice result must carry it alone
 * (wala/ML#957): before, the wala/ML#405 subscript pin and the wala/ML#953 feed both wrote that
 * destination, the pin its seed-time unknown dtype and the feed the int64, and the slice read
 * {@code {(2, 3) unknown, (2, 3) int64}}.
 */
public class TestSidecarSlice extends AbstractTensorTest {

  private static final String[] FILES = {
    "musictx_sidecar_proj/data.py", "musictx_sidecar_proj/driver.py"
  };

  private static final String PROJECT = "musictx_sidecar_proj";

  /** The anchored local transits {@code _get_seq}'s return: {@code (n,)} int64. */
  @Test
  public void testSidecarSequence()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "consume_seq",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new SymbolicDim("n"))))));
  }

  /** {@code np.array} over the comprehension of anchored results: {@code (2, 4)} int64. */
  @Test
  public void testSidecarBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "consume_batch",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new NumericDim(2), new NumericDim(4))))));
  }

  /** The slice of the batch carries the fed int64 alone: {@code (2, 3)} int64, no unknown twin. */
  @Test
  public void testSidecarSliceCarriesTheFedDtypeAlone()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "driver.py",
        "consume_x",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new NumericDim(2), new NumericDim(3))))));
  }
}
