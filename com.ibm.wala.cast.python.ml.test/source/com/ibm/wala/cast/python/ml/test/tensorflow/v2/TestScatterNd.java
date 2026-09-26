package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for {@code tf.scatter_nd}: the dtype is that of {@code updates}, the shape is the {@code
 * shape} argument.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestScatterNd extends AbstractTensorTest {

  private static final String FILE = "tf2_test_scatter_nd.py";

  /**
   * An edge-count builder: float32 ones per edge scattered onto a per-node vector whose length is a
   * {@code tf.shape} piece of the node embeddings, so the result reads a float32 vector of that
   * node count.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCountsFromShapePiece()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_counts", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }

  /**
   * A literal shape and int32 updates read an int32 vector of the literal length.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testStaticShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_counts_static", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 6))));
  }

  /**
   * The gather over the counts keeps the counts' dtype: the chain an edge-count builder feeds its
   * message function through.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatheredCountsKeepDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_gathered", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }
}
