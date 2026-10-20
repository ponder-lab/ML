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
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for the tensor that the {@code tf.keras.Model} summary's {@code call} returns: what
 * applying an instance of a {@code tf.keras.Model} subclass that does not override {@code call}
 * gives, or calling {@code call} on a model directly (wala/ML#1128).
 *
 * <p>{@code Model.call} runs the graph from the {@code inputs} to the {@code outputs} a functional
 * model was built with, so the result is the {@link ModelCall} one. A model whose construction
 * demonstrably passes no {@code outputs} has no graph to run: its {@code call} raises {@code
 * NotImplementedError}, and the call gives no tensor. When the {@code outputs} cannot be read, or
 * the construction cannot be found, the result is still a tensor, of the shape {@link ModelCall}
 * gives an unresolved model.
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
   * Whether some model this {@code call} may be called on can be a functional one: a {@code
   * tf.keras.Model} built directly, or a subclass instance whose construction does not demonstrably
   * omit {@code outputs}.
   *
   * @param builder The propagation call graph builder.
   * @return {@code false} only if every instance's construction demonstrably passes no {@code
   *     outputs}.
   */
  private boolean isFunctional(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());
    if (selfPts == null || selfPts.isEmpty()) return true;
    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);
      if (selfASIN != null
          && selfASIN.concreteType().getReference().equals(MODEL.getDeclaringClass())) return true;
      if (!getConstructionOutputs(builder, selfIK).omitted()) return true;
    }
    return false;
  }
}
