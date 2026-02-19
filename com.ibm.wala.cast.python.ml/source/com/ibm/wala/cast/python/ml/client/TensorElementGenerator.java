package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for elements of a tensor, used during iteration.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorElementGenerator extends TensorGenerator {

  private final TensorGenerator containerGenerator;

  public TensorElementGenerator(TensorGenerator containerGenerator) {
    super(containerGenerator.getSource());
    this.containerGenerator = containerGenerator;
  }

  /**
   * Returns the shapes of the elements contained within the container tensor.
   *
   * <p>When iterating over a tensor of rank {@code N}, the elements produced have rank {@code N-1},
   * where the first dimension of the container is removed. For example, iterating over a tensor of
   * shape {@code [5, 10]} yields elements of shape {@code [10]}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A set of possible element shapes.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> containerShapes = this.containerGenerator.getShapes(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : containerShapes) {
      if (shape.size() > 0) {
        // Peel off the first dimension.
        ret.add(new ArrayList<>(shape.subList(1, shape.size())));
      } else {
        // Iterating a scalar tensor is technically an error in TensorFlow, but we treat its
        // elements
        // as scalars for robustness.
        ret.add(Collections.emptyList());
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.containerGenerator.getDTypes(builder);
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
