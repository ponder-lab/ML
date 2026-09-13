package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Calls on {@code tf.keras.applications} models (wala/ML#896): the output rank is fixed by the
 * constructor's {@code include_top} and {@code pooling}, the batch axis is the input's, and the
 * architecture's own extents stay unresolved.
 */
public class TestKerasApplications extends AbstractTensorTest {

  private static final Dimension<?> U = UnresolvedDim.INSTANCE;

  private static final String BACKBONE = "tf2_test_keras_application_backbone.py";

  private static final String VARIANTS = "tf2_test_keras_application_variants.py";

  /**
   * A multi-GPU training script at its subject shape: a {@code MobileNetV2} backbone with {@code
   * include_top=False} inside a Functional model, wrapped by a {@code Sequential} with a softmax
   * {@code Dense(NUM_CLASS)} head, fed from {@code flow_from_directory} through {@code
   * strategy.run}. The backbone's rank-4 feature map is what lets {@code Flatten} then {@code
   * Dense} reach the head's {@code (None, 10)}; before the backbone was modeled its call result was
   * not a tensor at all and the prediction reached the loss with a dtype and no rank.
   *
   * <p>The fixture's own asserts see the generator's concrete batch (8, then a partial 4) inside
   * the traced step, two extents on one axis, which is what the analysis's {@code Dynamic} batch
   * axis (the {@code flow_from_directory} model's, wala/ML#830) claims.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBackboneInModel()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        BACKBONE,
        "consume_predictions",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(10))))));
    test(
        BACKBONE,
        "consume_backbone",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, U, U, U)))));
  }

  /**
   * The values beside the backbone that the model must not touch: the images the generator yields
   * and the labels, whose class-count axis the generator cannot see and a downstream loss fixes
   * (wala/ML#920).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBackboneControls()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        BACKBONE,
        "consume_images",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        DynamicDim.INSTANCE,
                        new NumericDim(112),
                        new NumericDim(112),
                        new NumericDim(3))))));
    // The labels' class axis is the directory's class count, which the generator cannot see; it is
    // fixed by the `CategoricalCrossentropy` call the labels reach beside the `(None, 10)`
    // predictions (wala/ML#920), so it reads 10 here rather than unresolved.
    test(
        BACKBONE,
        "consume_labels",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(10))))));
  }

  /**
   * The rank rule's branches on a concrete {@code (4, 224, 224, 3)} input: the default {@code
   * include_top} classifies to rank 2, {@code pooling} of {@code "avg"} or {@code "max"} reduces to
   * rank 2 (through the module path and the package path alike), {@code include_top=False} without
   * pooling keeps the rank-4 feature map, an {@code include_top} the program decides at runtime
   * declines to unknown rank rather than pick one, two constructions of different rank reaching one
   * call read as the union of both (each receiver instance dispatches on its own), one construction
   * site looping over both {@code include_top} values declines, a {@code pooling} read from the
   * environment declines, and an input the analysis has no shape for still gets the constructor's
   * rank with an unresolved batch axis, since the rank never depended on the input.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testVariants()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Set<TensorType> vector = Set.of(new TensorType(FLOAT_32, asList(new NumericDim(4), U)));
    test(VARIANTS, "consume_top", 1, 1, Map.of(2, vector));
    test(VARIANTS, "consume_avg", 1, 1, Map.of(2, vector));
    test(VARIANTS, "consume_max", 1, 1, Map.of(2, vector));
    test(
        VARIANTS,
        "consume_features",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, asList(new NumericDim(4), U, U, U)))));
    Set<TensorType> unknownRank = Set.of(new TensorType(FLOAT_32, null));
    test(VARIANTS, "consume_flag", 1, 1, Map.of(2, unknownRank));
    // Two constructions reaching one call dispatch per receiver instance, so the sink sees both
    // ranks; one construction site holding both `include_top` values cannot be split and declines.
    test(
        VARIANTS,
        "consume_disagree",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(4), U)),
                new TensorType(FLOAT_32, asList(new NumericDim(4), U, U, U)))));
    test(VARIANTS, "consume_looped", 1, 1, Map.of(2, unknownRank));
    // A pooling read from the environment is supplied but unreadable, so it declines; an input the
    // analysis has no shape for still gets the constructor's rank, with the batch unresolved.
    test(VARIANTS, "consume_envpool", 1, 1, Map.of(2, unknownRank));
    test(
        VARIANTS,
        "consume_opaque",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, asList(U, U, U, U)))));
  }
}
