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
 * A generator for ndarrays returned by {@code tf.keras.datasets.boston_housing.load_data()}.
 *
 * <p>The four ndarrays have well-known shapes and dtype {@code float64}:
 *
 * <ul>
 *   <li>{@code x_train}: {@code (404, 13)} — 13 numeric features per training example.
 *   <li>{@code y_train}: {@code (404,)} — median home values (regression target).
 *   <li>{@code x_test}: {@code (102, 13)}.
 *   <li>{@code y_test}: {@code (102,)}.
 * </ul>
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing/load_data">tf.keras.datasets.boston_housing.load_data</a>
 */
public class BostonHousingInputData extends TensorGenerator {

  private final List<Dimension<?>> shape;

  /**
   * Constructs from a points-to source.
   *
   * @param source The {@link PointsToSetVariable} representing the tensor.
   * @param shape The Boston-Housing shape — one of {@link #X_TRAIN_SHAPE}, {@link #Y_TRAIN_SHAPE},
   *     {@link #X_TEST_SHAPE}, {@link #Y_TEST_SHAPE}.
   */
  public BostonHousingInputData(PointsToSetVariable source, List<Dimension<?>> shape) {
    super(source);
    this.shape = shape;
  }

  /**
   * Constructs from a CG node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   * @param shape The Boston-Housing shape to report.
   */
  public BostonHousingInputData(CGNode node, List<Dimension<?>> shape) {
    super(node);
    this.shape = shape;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(shape);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.FLOAT64);
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

  /** Number of training examples. */
  public static final int NUM_TRAIN_EXAMPLES = 404;

  /** Number of test examples. */
  public static final int NUM_TEST_EXAMPLES = 102;

  /** Number of features per example. */
  public static final int NUM_FEATURES = 13;

  /** Features shape: {@code (N, 13)}. */
  public static List<Dimension<?>> featuresShape(int n) {
    return List.of(new NumericDim(n), new NumericDim(NUM_FEATURES));
  }

  /** Targets shape: {@code (N,)}. */
  public static List<Dimension<?>> targetsShape(int n) {
    return List.of(new NumericDim(n));
  }

  /** Shape of {@code boston_housing.load_data()[0][0]}: {@code (404, 13)} of {@code float64}. */
  public static final List<Dimension<?>> X_TRAIN_SHAPE = featuresShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code boston_housing.load_data()[0][1]}: {@code (404,)} of {@code float64}. */
  public static final List<Dimension<?>> Y_TRAIN_SHAPE = targetsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code boston_housing.load_data()[1][0]}: {@code (102, 13)} of {@code float64}. */
  public static final List<Dimension<?>> X_TEST_SHAPE = featuresShape(NUM_TEST_EXAMPLES);

  /** Shape of {@code boston_housing.load_data()[1][1]}: {@code (102,)} of {@code float64}. */
  public static final List<Dimension<?>> Y_TEST_SHAPE = targetsShape(NUM_TEST_EXAMPLES);

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
