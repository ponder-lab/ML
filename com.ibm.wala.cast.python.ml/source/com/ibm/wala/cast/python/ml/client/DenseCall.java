package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for invocations of {@code tf.keras.layers.Dense}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense">tf.keras.layers.Dense</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu>">Raffi Khatchadourian</a>
 */
public class DenseCall extends TensorGenerator {

  /**
   * Parameter positions and names for calls to {@code tf.keras.layers.Dense}.
   *
   * @see <a
   *     href="https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/layers/core/dense.py#L149">Dense.call</a>
   */
  protected enum Parameters {
    /** The input tensors. */
    INPUTS;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DenseCall(PointsToSetVariable source) {
    super(source);
  }

  public DenseCall(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    LOGGER.info("Deriving shape for Dense call at: " + this.getNode());
    // TODO
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Derive dtype from the input tensor.
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
    if (inputPts.isEmpty()) return EnumSet.noneOf(DType.class);
    return this.getDTypesOfValue(builder, inputPts);
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
