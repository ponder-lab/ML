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
 * A generator for ndarrays returned by {@code tf.keras.datasets.reuters.load_data()}.
 *
 * <p>The four ndarrays have the following shapes and dtypes:
 *
 * <ul>
 *   <li>{@code x_train}: shape {@code (8982,)}, dtype {@code object} — each element is a Python
 *       list of variable-length integer-encoded newswires (numpy {@code object} dtype).
 *   <li>{@code y_train}: shape {@code (8982,)}, dtype {@link DType#INT64} — topic labels.
 *   <li>{@code x_test}: shape {@code (2246,)}, dtype {@code object}.
 *   <li>{@code y_test}: shape {@code (2246,)}, dtype {@link DType#INT64}.
 * </ul>
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/reuters/load_data">tf.keras.datasets.reuters.load_data</a>
 */
public class ReutersInputData extends TensorGenerator {

  private final List<Dimension<?>> shape;

  private final Set<DType> dtypes;

  /**
   * Constructs from a points-to source.
   *
   * @param source The {@link PointsToSetVariable} representing the tensor.
   * @param shape The Reuters shape — one of {@link #X_TRAIN_SHAPE}, {@link #Y_TRAIN_SHAPE}, {@link
   *     #X_TEST_SHAPE}, {@link #Y_TEST_SHAPE}.
   * @param dtypes The dtype set: {@code int64} for labels, {@code object} for the
   *     variable-length-sequence arrays.
   */
  public ReutersInputData(PointsToSetVariable source, List<Dimension<?>> shape, Set<DType> dtypes) {
    super(source);
    this.shape = shape;
    this.dtypes = dtypes;
  }

  /**
   * Constructs from a CG node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   * @param shape The Reuters shape to report.
   * @param dtypes The dtype set.
   */
  public ReutersInputData(CGNode node, List<Dimension<?>> shape, Set<DType> dtypes) {
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

  /** Number of training newswires. */
  public static final int NUM_TRAIN_EXAMPLES = 8982;

  /** Number of test newswires. */
  public static final int NUM_TEST_EXAMPLES = 2246;

  /** Newswires shape: {@code (N,)} (variable-length sequences). */
  public static List<Dimension<?>> newswiresShape(int n) {
    return List.of(new NumericDim(n));
  }

  /** Labels shape: {@code (N,)} of {@code int64}. */
  public static List<Dimension<?>> labelsShape(int n) {
    return List.of(new NumericDim(n));
  }

  /** Shape of {@code reuters.load_data()[0][0]}: {@code (8982,)} of {@code object} dtype. */
  public static final List<Dimension<?>> X_TRAIN_SHAPE = newswiresShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code reuters.load_data()[0][1]}: {@code (8982,)} of {@code int64}. */
  public static final List<Dimension<?>> Y_TRAIN_SHAPE = labelsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code reuters.load_data()[1][0]}: {@code (2246,)} of {@code object} dtype. */
  public static final List<Dimension<?>> X_TEST_SHAPE = newswiresShape(NUM_TEST_EXAMPLES);

  /** Shape of {@code reuters.load_data()[1][1]}: {@code (2246,)} of {@code int64}. */
  public static final List<Dimension<?>> Y_TEST_SHAPE = labelsShape(NUM_TEST_EXAMPLES);

  /**
   * Dtype set for the variable-length-sequence {@code x_*} arrays: numpy {@code object}
   * (wala/ML#488).
   */
  public static final Set<DType> X_DTYPES = EnumSet.of(DType.OBJECT);

  /** Dtype set for the topic-label {@code y_*} arrays. */
  public static final Set<DType> Y_DTYPES = EnumSet.of(DType.INT64);
}
