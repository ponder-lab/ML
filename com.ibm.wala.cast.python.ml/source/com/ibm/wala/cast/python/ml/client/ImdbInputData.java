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
 * A generator for ndarrays returned by {@code tf.keras.datasets.imdb.load_data()}.
 *
 * <p>The four ndarrays have the following shapes and dtypes:
 *
 * <ul>
 *   <li>{@code x_train} / {@code x_test}: shape {@code (25000,)}, dtype unknown — each element is a
 *       Python list of variable-length integer-encoded reviews (numpy {@code object} dtype at
 *       runtime, not directly representable as a regular tensor until padded).
 *   <li>{@code y_train} / {@code y_test}: shape {@code (25000,)}, dtype {@link DType#INT64} —
 *       binary labels (0 or 1).
 * </ul>
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data">tf.keras.datasets.imdb.load_data</a>
 */
public class ImdbInputData extends TensorGenerator {

  private final List<Dimension<?>> shape;

  private final Set<DType> dtypes;

  /**
   * Constructs from a points-to source.
   *
   * @param source The {@link PointsToSetVariable} representing the tensor.
   * @param shape The IMDB shape to report — one of {@link #X_TRAIN_SHAPE}, {@link #Y_TRAIN_SHAPE},
   *     {@link #X_TEST_SHAPE}, {@link #Y_TEST_SHAPE}.
   * @param dtypes The dtype set: {@code int64} for the {@code y_*} arrays, {@code unknown} for the
   *     {@code x_*} arrays (which are not regular tensors).
   */
  public ImdbInputData(PointsToSetVariable source, List<Dimension<?>> shape, Set<DType> dtypes) {
    super(source);
    this.shape = shape;
    this.dtypes = dtypes;
  }

  /**
   * Constructs from a CG node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   * @param shape The IMDB shape to report.
   * @param dtypes The dtype set.
   */
  public ImdbInputData(CGNode node, List<Dimension<?>> shape, Set<DType> dtypes) {
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

  /** Number of training reviews in IMDB's {@code load_data()} output. */
  public static final int NUM_TRAIN_EXAMPLES = 25000;

  /** Number of test reviews in IMDB's {@code load_data()} output. */
  public static final int NUM_TEST_EXAMPLES = 25000;

  /** Reviews shape: {@code (N,)} (each element is a variable-length sequence). */
  public static List<Dimension<?>> reviewsShape(int n) {
    return List.of(new NumericDim(n));
  }

  /** Labels shape: {@code (N,)} of {@code int64}. */
  public static List<Dimension<?>> labelsShape(int n) {
    return List.of(new NumericDim(n));
  }

  /** Shape of {@code imdb.load_data()[0][0]}: {@code (25000,)} of unknown dtype. */
  public static final List<Dimension<?>> X_TRAIN_SHAPE = reviewsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code imdb.load_data()[0][1]}: {@code (25000,)} of {@code int64}. */
  public static final List<Dimension<?>> Y_TRAIN_SHAPE = labelsShape(NUM_TRAIN_EXAMPLES);

  /** Shape of {@code imdb.load_data()[1][0]}: {@code (25000,)} of unknown dtype. */
  public static final List<Dimension<?>> X_TEST_SHAPE = reviewsShape(NUM_TEST_EXAMPLES);

  /** Shape of {@code imdb.load_data()[1][1]}: {@code (25000,)} of {@code int64}. */
  public static final List<Dimension<?>> Y_TEST_SHAPE = labelsShape(NUM_TEST_EXAMPLES);

  /** Dtype set for the variable-length-sequence {@code x_*} arrays. */
  public static final Set<DType> X_DTYPES = EnumSet.of(DType.UNKNOWN);

  /** Dtype set for the binary-label {@code y_*} arrays. */
  public static final Set<DType> Y_DTYPES = EnumSet.of(DType.INT64);
}
