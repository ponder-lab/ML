package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A layer that reshapes its result to a shape list built from {@code tf.shape(inputs)} subscripts
 * and a stored attribute, {@code [tf.shape(inputs)[0], tf.shape(inputs)[1]] + [self.filter_size]},
 * keeps its input's rank (wala/ML#987), in every instance and through a chain of two such layers,
 * and a {@code tf.split} over such a result keeps it as well (wala/ML#986). Before the fix the
 * reshape's own target resolved, but the dataflow's reshape node op pinned the result at an unknown
 * rank, since the shape argument is neither a literal container nor a recognized shape-vector
 * chain, and that pin replaced every member the resolved target produced.
 */
public class TestConv1dReshapeRank extends AbstractTensorTest {

  private static final String FILE = "tf2_test_conv1d_reshape_rank.py";

  /** The first layer's result: rank 3 with the layer's own filter size, and no rankless twin. */
  @Test
  public void testFirstLayerKeepsRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5, 32))));
  }

  /** The second layer, fed by the first: the chain keeps the rank across both. */
  @Test
  public void testChainedLayerKeepsRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_second", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5, 8))));
  }

  /**
   * The control for the node op's widened consult: a shape operand opaque to every reader, the
   * value of an unmodeled call with no points-to set, still reads as a tensor of unknown rank, the
   * pin's own reading, before and after; without the points-to guard on the consult, the
   * generator's input-shape fallback would replace it with the input's shape.
   */
  @Test
  public void testOpaqueShapeStaysUnknownRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_opaque", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A layer whose filter size the analysis cannot compute (a configuration value times three, the
   * attention projection's idiom) keeps its rank, with an unresolved last axis (wala/ML#986):
   * before, the unresolvable element emptied the whole target and the result was rankless.
   */
  @Test
  public void testUnresolvableFilterKeepsRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_wide",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(2), new NumericDim(5), UnresolvedDim.INSTANCE)))));
  }

  /** A {@code tf.split} over that result: each piece keeps the rank, its last axis unresolved. */
  @Test
  public void testSplitOfUnresolvableFilterKeepsRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_wide_split",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(2), new NumericDim(5), UnresolvedDim.INSTANCE)))));
  }

  /**
   * A shape element that is arithmetic over stored attributes, one of which the analysis cannot
   * compute ({@code self.wide_size // self.n_head}, a transformer's head size), reads as an
   * unresolved axis, so the reshape keeps its rank (wala/ML#986); the computable attribute beside
   * it reads as its value.
   */
  @Test
  public void testArithmeticOverAttributesKeepsRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_heads",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(2),
                        new NumericDim(5),
                        new NumericDim(3),
                        UnresolvedDim.INSTANCE)))));
  }

  /** A {@code tf.split} over a third instance's result: each piece keeps the rank. */
  @Test
  public void testSplitOfLayerResultKeepsRank()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_split", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5, 8))));
  }
}
