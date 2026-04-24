package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for ndarrays returned by {@code tf.keras.datasets.cifar10.load_data()}.
 *
 * <p>Each of the four ndarrays ({@code x_train}, {@code y_train}, {@code x_test}, {@code y_test})
 * has a fixed, well-known shape and dtype ({@code uint8}). The {@code x_*} arrays are 4-D RGB
 * images {@code (N, 32, 32, 3)}; the {@code y_*} arrays are 2-D label columns {@code (N, 1)}.
 *
 * <p>Mirrors {@link MnistInputData} but with CIFAR-10-specific shapes; kept as a sibling class
 * rather than a shared parameterized base because the two datasets have different image dimensions
 * and label ranks, and the sibling-class approach keeps the factory dispatch in {@link
 * TensorGeneratorFactory} symmetrical and easy to read.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data">tf.keras.datasets.cifar10.load_data</a>
 */
public class Cifar10InputData extends TensorGenerator {

  private final List<Dimension<?>> shape;

  /**
   * Constructs a CIFAR-10 input data generator bound to an existing points-to source.
   *
   * @param source The points-to set variable representing the tensor whose shape/dtype this
   *     generator produces.
   * @param shape The CIFAR-10 shape to report for {@code source}.
   */
  public Cifar10InputData(PointsToSetVariable source, List<Dimension<?>> shape) {
    super(source);
    this.shape = shape;
  }

  /**
   * Constructs a CIFAR-10 input data generator bound to a call-graph node (manual-generator path,
   * used when no points-to source is available).
   *
   * @param node The CG node the generator is anchored at.
   * @param shape The CIFAR-10 shape to report for the node's tensor.
   */
  public Cifar10InputData(CGNode node, List<Dimension<?>> shape) {
    super(node);
    this.shape = shape;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(shape);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UINT8);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  /** Number of training examples in CIFAR-10's {@code load_data()} output. */
  public static final int NUM_TRAIN_EXAMPLES = 50000;

  /** Number of test examples in CIFAR-10's {@code load_data()} output. */
  public static final int NUM_TEST_EXAMPLES = 10000;

  /** Side length (height and width) of each CIFAR-10 image, in pixels. */
  public static final int IMAGE_SIDE = 32;

  /** Number of color channels per CIFAR-10 pixel (RGB). */
  public static final int NUM_CHANNELS = 3;

  /** CIFAR-10 training/test images: {@code (N, 32, 32, 3)} of {@code uint8}. */
  public static List<Dimension<?>> imagesShape(int n) {
    return List.of(
        new NumericDim(n),
        new NumericDim(IMAGE_SIDE),
        new NumericDim(IMAGE_SIDE),
        new NumericDim(NUM_CHANNELS));
  }

  /** CIFAR-10 training/test labels: {@code (N, 1)} of {@code uint8}. */
  public static List<Dimension<?>> labelsShape(int n) {
    return List.of(new NumericDim(n), new NumericDim(1));
  }

  /** Shape of {@code cifar10.load_data()[0][0]}: {@code (50000, 32, 32, 3)} of {@code uint8}. */
  public static final List<Dimension<?>> X_TRAIN_SHAPE = imagesShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code cifar10.load_data()[0][1]}: {@code (50000, 1)} of {@code uint8}. */
  public static final List<Dimension<?>> Y_TRAIN_SHAPE = labelsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code cifar10.load_data()[1][0]}: {@code (10000, 32, 32, 3)} of {@code uint8}. */
  public static final List<Dimension<?>> X_TEST_SHAPE = imagesShape(NUM_TEST_EXAMPLES);

  /** Shape of {@code cifar10.load_data()[1][1]}: {@code (10000, 1)} of {@code uint8}. */
  public static final List<Dimension<?>> Y_TEST_SHAPE = labelsShape(NUM_TEST_EXAMPLES);
}
