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

  private final List<Dimension<?>> shape;
  private final DType dtype;

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
