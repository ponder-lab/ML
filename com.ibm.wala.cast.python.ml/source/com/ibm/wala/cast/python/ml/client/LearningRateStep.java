package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

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
 * The step an optimizer passes to its learning-rate schedule (<a
 * href="https://github.com/wala/ML/issues/1050">wala/ML#1050</a>): a scalar {@code float32}.
 *
 * <p>Keras 2.9's {@code OptimizerV2._decayed_lr} calls the schedule as {@code
 * lr_t(tf.cast(self.iterations, var_dtype))}, the optimizer's step count cast to the dtype of the
 * variable it updates. That dtype is {@code float32} unless the program trains in a lower
 * precision, which this does not model.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule">tf.keras.optimizers.schedules.LearningRateSchedule</a>
 */
public class LearningRateStep extends TensorGenerator {

  /**
   * Constructs from a points-to source.
   *
   * @param source The {@link PointsToSetVariable} representing the step.
   */
  public LearningRateStep(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs from a CG node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method allocating the step.
   */
  public LearningRateStep(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(Collections.emptyList());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(FLOAT32);
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
