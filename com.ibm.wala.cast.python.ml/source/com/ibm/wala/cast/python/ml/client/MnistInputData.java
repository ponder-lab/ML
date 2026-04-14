package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * A generator for one of the four arrays returned by {@code tf.keras.datasets.mnist.load_data()}.
 * Each instance is constructed with a hard-coded shape and dtype — {@code (60000, 28, 28) uint8}
 * for {@code x_train}, {@code (60000,) uint8} for {@code y_train}, and the corresponding {@code
 * 10000}-row variants for the test split.
 *
 * <p>See wala/ML#361. The four allocation classes that this generator is dispatched for ({@code
 * MNIST_X_TRAIN_TYPE}, {@code MNIST_Y_TRAIN_TYPE}, {@code MNIST_X_TEST_TYPE}, {@code
 * MNIST_Y_TEST_TYPE}) are declared in {@link com.ibm.wala.cast.python.ml.types.TensorFlowTypes} and
 * used in {@code tensorflow.xml}'s {@code load_data.do} method body.
 */
public class MnistInputData extends TensorGenerator {

  /**
   * The hard-coded shape of this mnist array — e.g., {@code (60000, 28, 28)} for {@code x_train},
   * {@code (10000,)} for {@code y_test}. Supplied by {@link
   * com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory} at dispatch time based on which of
   * the four mnist allocation classes matched.
   */
  private final List<Dimension<?>> shape;

  /**
   * The hard-coded element dtype of this mnist array. Always {@link DType#UINT8} in practice — both
   * the images and labels returned by {@code tf.keras.datasets.mnist.load_data()} are uint8 at
   * runtime — but stored as a field for symmetry with {@link #shape} and to leave room for future
   * dataset loaders whose elements have different dtypes.
   */
  private final DType dtype;

  /**
   * Constructs a generator for one of the four arrays returned by {@code
   * tf.keras.datasets.mnist.load_data()}, parameterized with the element's known shape and dtype.
   *
   * @param source the {@link PointsToSetVariable} representing the mnist allocation site this
   *     generator is dispatched for — typically the {@code x_train}, {@code y_train}, {@code
   *     x_test}, or {@code y_test} new-instance declared in {@code tensorflow.xml}'s {@code
   *     load_data.do} body.
   * @param shape the hard-coded shape for this element — e.g., {@code (60000, 28, 28)} for {@code
   *     x_train}. Never {@code null}.
   * @param dtype the hard-coded dtype for this element — always {@link DType#UINT8} in practice.
   *     Never {@code null}.
   */
  public MnistInputData(PointsToSetVariable source, List<Dimension<?>> shape, DType dtype) {
    super(source);
    this.shape = shape;
    this.dtype = dtype;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Set.of(shape);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(dtype);
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
}
