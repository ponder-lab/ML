package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
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
 * A generator for ndarrays returned by {@code tf.keras.datasets.cifar100.load_data()}.
 *
 * <p>Shapes mirror {@link Cifar10InputData} exactly: image arrays are 4-D RGB images {@code (N, 32,
 * 32, 3)}; label arrays are 2-D label columns {@code (N, 1)}. The dtype, however, diverges from
 * cifar10:
 *
 * <ul>
 *   <li>{@code x_train} / {@code x_test}: {@link DType#UINT8} (same as cifar10).
 *   <li>{@code y_train} / {@code y_test}: {@link DType#INT64} &mdash; cifar100's labels are 64-bit
 *       integer class indices, NOT {@link DType#UINT8} as for cifar10. This is the key reason
 *       cifar100 has its own generator class instead of reusing {@code Cifar10InputData}'s
 *       hardcoded uint8 dtype.
 * </ul>
 *
 * Per-array shape and dtype are passed in by the dispatch site in {@link TensorGeneratorFactory}
 * (paired through the four {@code CIFAR100_*} {@link
 * com.ibm.wala.cast.python.ml.types.TensorFlowTypes} arms) so a single class handles all four
 * arrays.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100/load_data">tf.keras.datasets.cifar100.load_data</a>
 * @see <a href="https://github.com/wala/ML/issues/487">wala/ML#487</a>
 */
public class Cifar100InputData extends TensorGenerator {

  private final List<Dimension<?>> shape;

  private final Set<DType> dtypes;

  /**
   * Constructs from a points-to source.
   *
   * @param source The {@link PointsToSetVariable} representing the tensor.
   * @param shape The cifar100 shape &mdash; one of {@link #X_TRAIN_SHAPE}, {@link #Y_TRAIN_SHAPE},
   *     {@link #X_TEST_SHAPE}, {@link #Y_TEST_SHAPE}.
   * @param dtypes The dtype set: {@link #X_DTYPES} ({@code uint8}) for image arrays, {@link
   *     #Y_DTYPES} ({@code int64}) for label arrays.
   */
  public Cifar100InputData(
      PointsToSetVariable source, List<Dimension<?>> shape, Set<DType> dtypes) {
    super(source);
    this.shape = shape;
    this.dtypes = dtypes;
  }

  /**
   * Constructs from a CG node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   * @param shape The cifar100 shape to report.
   * @param dtypes The dtype set.
   */
  public Cifar100InputData(CGNode node, List<Dimension<?>> shape, Set<DType> dtypes) {
    super(node);
    this.shape = shape;
    this.dtypes = dtypes;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(shape);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return dtypes;
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

  /** Number of training examples in cifar100's {@code load_data()} output. */
  public static final int NUM_TRAIN_EXAMPLES = 50000;

  /** Number of test examples in cifar100's {@code load_data()} output. */
  public static final int NUM_TEST_EXAMPLES = 10000;

  /** Side length (height and width) of each cifar100 image, in pixels. */
  public static final int IMAGE_SIDE = 32;

  /** Number of color channels per cifar100 pixel (RGB). */
  public static final int NUM_CHANNELS = 3;

  /** Cifar100 training/test images: {@code (N, 32, 32, 3)} of {@code uint8}. */
  public static List<Dimension<?>> imagesShape(int n) {
    return List.of(
        new NumericDim(n),
        new NumericDim(IMAGE_SIDE),
        new NumericDim(IMAGE_SIDE),
        new NumericDim(NUM_CHANNELS));
  }

  /** Cifar100 training/test labels: {@code (N, 1)} of {@code int64}. */
  public static List<Dimension<?>> labelsShape(int n) {
    return List.of(new NumericDim(n), new NumericDim(1));
  }

  /** Shape of {@code cifar100.load_data()[0][0]}: {@code (50000, 32, 32, 3)} of {@code uint8}. */
  public static final List<Dimension<?>> X_TRAIN_SHAPE = imagesShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code cifar100.load_data()[0][1]}: {@code (50000, 1)} of {@code int64}. */
  public static final List<Dimension<?>> Y_TRAIN_SHAPE = labelsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code cifar100.load_data()[1][0]}: {@code (10000, 32, 32, 3)} of {@code uint8}. */
  public static final List<Dimension<?>> X_TEST_SHAPE = imagesShape(NUM_TEST_EXAMPLES);

  /** Shape of {@code cifar100.load_data()[1][1]}: {@code (10000, 1)} of {@code int64}. */
  public static final List<Dimension<?>> Y_TEST_SHAPE = labelsShape(NUM_TEST_EXAMPLES);

  /** Dtype set for cifar100's {@code x_*} image arrays: {@code uint8}. */
  public static final Set<DType> X_DTYPES = EnumSet.of(DType.UINT8);

  /** Dtype set for cifar100's {@code y_*} label arrays: {@code int64}. */
  public static final Set<DType> Y_DTYPES = EnumSet.of(DType.INT64);

  /**
   * Returns the producing library of the modeled value: {@code load_data} returns numpy arrays, not
   * tensors, so the value is an ndarray (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
