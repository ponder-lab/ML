package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a {@code tf.keras.Sequential} model's call calls the user layers in its list, so each
 * layer's {@code call} receives its input through the model (<a
 * href="https://github.com/wala/ML/issues/832">wala/ML#832</a>). Every layer class is called only
 * through its model, so its parameter's type comes from that path alone. The parameter is read in
 * the layer's own {@code call}, with no sink.
 */
public class TestSequentialLayerCall extends AbstractTensorTest {

  private static final String FILE = "tf2_test_sequential_layer_call.py";

  private static final String STAGES_FILE = "tf2_test_sequential_layer_call_stages.py";

  private static final TensorType TENSOR_2_50_16_FLOAT32 = TensorType.of(FLOAT_32, 2, 50, 16);

  private static final TensorType TENSOR_3_40_80_FLOAT32 = TensorType.of(FLOAT_32, 3, 40, 80);

  private static final TensorType TENSOR_3_40_16_FLOAT32 = TensorType.of(FLOAT_32, 3, 40, 16);

  private static final TensorType TENSOR_3_40_8_FLOAT32 = TensorType.of(FLOAT_32, 3, 40, 8);

  private static final TensorType TENSOR_4_30_80_FLOAT32 = TensorType.of(FLOAT_32, 4, 30, 80);

  private static final TensorType TENSOR_4_30_16_FLOAT32 = TensorType.of(FLOAT_32, 4, 30, 16);

  private static final TensorType TENSOR_5_20_80_FLOAT32 = TensorType.of(FLOAT_32, 5, 20, 80);

  private static final TensorType TENSOR_5_20_16_FLOAT32 = TensorType.of(FLOAT_32, 5, 20, 16);

  private static final TensorType TENSOR_5_20_8_FLOAT32 = TensorType.of(FLOAT_32, 5, 20, 8);

  /**
   * A list written out: the layer at position 1 is called on the output of the layer at position 0,
   * exactly.
   */
  @Test
  public void testLiteral() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "UpLiteral.call", 1, 2, Map.of(3, Set.of(TENSOR_2_50_16_FLOAT32)));
  }

  /**
   * A list built by {@code +=} keeps no positions, so its layers are called on the join of the
   * model's input and every layer's output, which contains the runtime input.
   */
  @Test
  public void testConcatenated() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "UpPlus.call",
        1,
        2,
        Map.of(3, Set.of(TENSOR_3_40_80_FLOAT32, TENSOR_3_40_16_FLOAT32, TENSOR_3_40_8_FLOAT32)));
  }

  /** As {@link #testConcatenated()}, with the list grown in a loop. */
  @Test
  public void testLoopConcatenated() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "UpLoop.call",
        1,
        2,
        Map.of(3, Set.of(TENSOR_4_30_80_FLOAT32, TENSOR_4_30_16_FLOAT32)));
  }

  /** A model built by {@code add}, whose order is not kept: the layers are called on the join. */
  @Test
  public void testAdded() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "UpAdd.call",
        1,
        2,
        Map.of(3, Set.of(TENSOR_5_20_80_FLOAT32, TENSOR_5_20_16_FLOAT32, TENSOR_5_20_8_FLOAT32)));
  }

  /**
   * The layers' join, in a generator whose list grows by {@code +=} in nested loops and holds a
   * user layer per stage. Every member is a shape some layer of the list outputs, or the model's
   * input. It holds the runtime input of the first stage exactly, and the second stage's under
   * unresolved axes; the unknown-rank member comes from a layer whose output is not resolved.
   */
  private static final Set<TensorType> STAGES_JOIN =
      Set.of(
          TensorType.of(FLOAT_32, 2, 10, 80),
          TensorType.of(FLOAT_32, 2, 16, 80),
          TENSOR_UNKNOWN_SHAPE_FLOAT32,
          shape(2, 10, null),
          shape(2, null, null),
          shape(2, 16, null),
          shape(2, 4, null),
          TensorType.of(FLOAT_32, 2, 10, 1),
          shape(2, null, 1),
          TensorType.of(FLOAT_32, 2, 4, 1),
          TensorType.of(FLOAT_32, 2, 10, 64),
          TensorType.of(FLOAT_32, 2, 4, 64),
          shape(2, null, 64));

  /**
   * A {@code float32} type whose {@code null} sizes are unresolved axes.
   *
   * @param sizes The axis sizes.
   * @return The type.
   */
  private static TensorType shape(Integer... sizes) {
    List<Dimension<?>> dims = new ArrayList<>();
    for (Integer size : sizes)
      dims.add(size == null ? UnresolvedDim.INSTANCE : new NumericDim(size));
    return new TensorType(FLOAT_32, dims);
  }

  /** The transposed-convolution layer of each upsampling stage receives the join. */
  @Test
  public void testStagesTranspose() throws ClassHierarchyException, CancelException, IOException {
    test(STAGES_FILE, "TFConvTranspose1d.call", 1, 4, Map.of(3, STAGES_JOIN));
  }

  /** The residual stacks after each stage receive the join too. */
  @Test
  public void testStagesStack() throws ClassHierarchyException, CancelException, IOException {
    test(STAGES_FILE, "TFResidualStack.call", 1, 6, Map.of(3, STAGES_JOIN));
  }

  /**
   * A layer at a literal position past the summary's unroll is not called, so its {@code call} is
   * not in the call graph.
   *
   * <p>TODO: Called once https://github.com/wala/ML/issues/1089 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testPastUnroll() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_sequential_layer_call_long.py",
        "Last.call",
        1,
        2,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }
}
