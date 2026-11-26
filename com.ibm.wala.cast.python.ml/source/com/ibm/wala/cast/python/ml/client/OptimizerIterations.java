package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * The step count a Keras {@code TensorBoard} callback passes to a learning-rate schedule (<a
 * href="https://github.com/wala/ML/issues/1341">wala/ML#1341</a>): a scalar {@code int64}.
 *
 * <p>Keras 2.9's {@code TensorBoard._collect_learning_rate} calls the schedule as {@code
 * lr_schedule(self.model.optimizer.iterations)}, with the optimizer's step count as it is. That
 * count is an {@code int64} variable, unlike the {@code float32} the optimizer passes when it
 * applies gradients ({@link LearningRateStep}).
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/callbacks/TensorBoard">tf.keras.callbacks.TensorBoard</a>
 */
public class OptimizerIterations extends TensorGenerator {

  /**
   * Constructs from a points-to source.
   *
   * @param source The {@link PointsToSetVariable} representing the step count.
   */
  protected OptimizerIterations(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs from a CG node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method allocating the step count.
   */
  protected OptimizerIterations(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(Collections.emptyList());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(INT64);
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
