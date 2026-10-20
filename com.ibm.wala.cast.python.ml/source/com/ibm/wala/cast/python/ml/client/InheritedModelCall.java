package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for the tensor that the {@code tf.keras.Model} summary's {@code call} returns: what
 * applying an instance of a {@code tf.keras.Model} subclass that does not override {@code call}
 * gives, or calling {@code call} on a model directly (wala/ML#1128).
 *
 * <p>{@code Model.call} runs the graph from the {@code inputs} to the {@code outputs} a functional
 * model was built with, so the result is the {@link ModelCall} one. A model built with no {@code
 * outputs}, a subclass whose constructor does not pass them to {@code super().__init__}, has no
 * graph to run: its {@code call} raises {@code NotImplementedError}, and the call gives no tensor.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/Model#call">
 *     tf.keras.Model.call</a>.
 */
public class InheritedModelCall extends ModelCall {

  public InheritedModelCall(PointsToSetVariable source) {
    super(source);
  }

  public InheritedModelCall(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.isFunctional(builder) ? super.getDefaultShapes(builder) : Collections.emptySet();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.isFunctional(builder) ? super.getDefaultDTypes(builder) : Collections.emptySet();
  }

  /**
   * Whether the model this {@code call} is called on was built with {@code outputs}: either it is a
   * {@code tf.keras.Model} built directly, or its constructor passes {@code outputs} to {@code
   * super().__init__}.
   *
   * @param builder The propagation call graph builder.
   * @return {@code true} if some instance the call may be on is a functional model.
   */
  private boolean isFunctional(PropagationCallGraphBuilder builder) {
    for (InstanceKey selfIK :
        this.getArgumentPointsToSet(
            builder, Parameters.SELF.getIndex(), Parameters.SELF.getName())) {
      AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);
      if (selfASIN != null
          && selfASIN.concreteType().getReference().equals(MODEL.getDeclaringClass())) return true;
      if (!getSuperInitOutputsPTS(builder, selfIK).isEmpty()) return true;
    }
    return false;
  }
}
