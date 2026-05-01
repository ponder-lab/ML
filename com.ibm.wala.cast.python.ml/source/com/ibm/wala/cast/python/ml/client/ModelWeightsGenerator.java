package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * A generator representing the weights (trainable or otherwise) of a Keras Model.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ModelWeightsGenerator extends TensorGenerator
    implements DelegatingTensorGenerator, TupleElementProvider {

  private final Model modelGenerator;

  public ModelWeightsGenerator(PointsToSetVariable source, Model modelGenerator) {
    super(source);
    this.modelGenerator = modelGenerator;
  }

  @Override
  public TensorGenerator getUnderlying() {
    return modelGenerator;
  }

  @Override
  public String toString() {
    return "ModelWeightsGenerator(" + modelGenerator + ")";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return modelGenerator.getWeightShapes(builder, this.getSource());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return modelGenerator.getWeightDTypes(builder, this.getSource());
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    // For simplicity, we treat all weights as having all possible weight types.
    // In Keras, weights are a list, so indexing is possible.
    return this.getTensorTypes(builder);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getShapes(builder);
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getDTypes(builder);
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
