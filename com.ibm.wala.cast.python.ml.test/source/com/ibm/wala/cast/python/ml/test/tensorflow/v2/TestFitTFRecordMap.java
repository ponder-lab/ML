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
}
