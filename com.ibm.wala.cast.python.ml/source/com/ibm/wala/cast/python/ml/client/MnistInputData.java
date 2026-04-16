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

  private final List<Dimension<?>> shape;

  public MnistInputData(PointsToSetVariable source, List<Dimension<?>> shape) {
    super(source);
    this.shape = shape;
  }

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

  /** MNIST training/test images: {@code (N, 28, 28)} of {@code uint8}. */
  public static List<Dimension<?>> imagesShape(int n) {
    return List.of(new NumericDim(n), new NumericDim(28), new NumericDim(28));
  }

  /** MNIST training/test labels: {@code (N,)} of {@code uint8}. */
  public static List<Dimension<?>> labelsShape(int n) {
    return List.of(new NumericDim(n));
  }
}
