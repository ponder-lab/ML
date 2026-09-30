package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * An edge list extended with self loops (wala/ML#985), on {@code tf2_test_concat_tile_rank.py}: the
 * self loops are {@code tf.tile(tf.expand_dims(tf.range(0, n), 1), [1, 2])} over a node count the
 * analysis cannot compute, and they are concatenated with a masked subscript of the edges whose own
 * shape does not resolve. {@code tf.tile} made its whole result ⊤ over the unresolved axis, and
 * {@code tf.concat} made its result ⊤ over the unresolved element, so the edge indices and the node
 * states gathered through them lost their ranks. {@code tf.tile} now keeps its input's rank, and
 * {@code tf.concat} takes its rank and non-axis dimensions from any element whose shape is known,
 * since every element must share them.
 */
public class TestConcatTileRank extends AbstractTensorTest {

  private static final String FILE = "tf2_test_concat_tile_rank.py";

  private static TensorType int32(Dimension<?>... dims) {
    return new TensorType(INT_32, List.of(dims));
  }

  /** The tiled self loops: the unresolved node count by {@code 2}. */
  @Test
  public void testTile() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_loops",
        1,
        1,
        Map.of(2, Set.of(int32(UnresolvedDim.INSTANCE, new NumericDim(2)))));
  }

  /**
   * The extended edge list: rank 2 from the loops, although the masked edges do not resolve. The
   * masked edges' extent is data-dependent, so the concatenated extent is dynamic.
   */
  @Test
  public void testConcat() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_edges",
        1,
        1,
        Map.of(2, Set.of(int32(DynamicDim.INSTANCE, new NumericDim(2)))));
  }

  /** The node states gathered through the edge targets, a helper's first parameter. */
  @Test
  public void testGatheredStates() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_src",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, List.of(DynamicDim.INSTANCE, new NumericDim(8))))));
  }

  /** The edge targets, a helper's second parameter. */
  @Test
  public void testEdgeTargets() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_index", 1, 1, Map.of(2, Set.of(int32(DynamicDim.INSTANCE))));
  }

  /** Control: elements of known but different ranks cannot be concatenated, so ⊤ stays. */
  @Test
  public void testRankMismatch() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mismatch", 1, 1, Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }
}
