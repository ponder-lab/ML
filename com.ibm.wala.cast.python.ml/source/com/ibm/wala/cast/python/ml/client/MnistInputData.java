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
 * A generator for ndarrays returned by {@code tf.keras.datasets.mnist.load_data()}.
 *
 * <p>Each of the four ndarrays ({@code x_train}, {@code y_train}, {@code x_test}, {@code y_test})
 * has a fixed, well-known shape and dtype ({@code uint8}).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data">tf.keras.datasets.mnist.load_data</a>
 */
public class MnistInputData extends TensorGenerator {

  /**
   * The MNIST shape this generator reports for its source. One of the four canonical shapes exposed
   * by {@code tf.keras.datasets.mnist.load_data()}: {@link #X_TRAIN_SHAPE} ({@code (60000, 28,
   * 28)}), {@link #Y_TRAIN_SHAPE} ({@code (60000,)}), {@link #X_TEST_SHAPE} ({@code (10000, 28,
   * 28)}), or {@link #Y_TEST_SHAPE} ({@code (10000,)}).
   */
  private final List<Dimension<?>> shape;

  /**
   * Constructs an MNIST input data generator bound to an existing points-to source.
   *
   * @param source The points-to set variable representing the tensor whose shape/dtype this
   *     generator produces.
   * @param shape The MNIST shape to report for {@code source}.
   */
  public MnistInputData(PointsToSetVariable source, List<Dimension<?>> shape) {
    super(source);
    this.shape = shape;
  }

  /**
   * Constructs an MNIST input data generator bound to a call-graph node (manual-generator path,
   * used when no points-to source is available).
   *
   * @param node The CG node the generator is anchored at.
   * @param shape The MNIST shape to report for the node's tensor.
   */
  public MnistInputData(CGNode node, List<Dimension<?>> shape) {
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

  /** Number of training examples in MNIST's {@code load_data()} output. */
  public static final int NUM_TRAIN_EXAMPLES = 60000;

  /** Number of test examples in MNIST's {@code load_data()} output. */
  public static final int NUM_TEST_EXAMPLES = 10000;

  /** Side length (height and width) of each MNIST image, in pixels. */
  public static final int IMAGE_SIDE = 28;

  /** MNIST training/test images: {@code (N, 28, 28)} of {@code uint8}. */
  public static List<Dimension<?>> imagesShape(int n) {
    return List.of(new NumericDim(n), new NumericDim(IMAGE_SIDE), new NumericDim(IMAGE_SIDE));
  }

  /** MNIST training/test labels: {@code (N,)} of {@code uint8}. */
  public static List<Dimension<?>> labelsShape(int n) {
    return List.of(new NumericDim(n));
  }

  /** Shape of {@code mnist.load_data()[0][0]}: {@code (60000, 28, 28)} of {@code uint8}. */
  public static final List<Dimension<?>> X_TRAIN_SHAPE = imagesShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code mnist.load_data()[0][1]}: {@code (60000,)} of {@code uint8}. */
  public static final List<Dimension<?>> Y_TRAIN_SHAPE = labelsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code mnist.load_data()[1][0]}: {@code (10000, 28, 28)} of {@code uint8}. */
  public static final List<Dimension<?>> X_TEST_SHAPE = imagesShape(NUM_TEST_EXAMPLES);

  /** Shape of {@code mnist.load_data()[1][1]}: {@code (10000,)} of {@code uint8}. */
  public static final List<Dimension<?>> Y_TEST_SHAPE = labelsShape(NUM_TEST_EXAMPLES);
}
