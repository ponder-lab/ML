package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator representing an iterator over a collection.
 *
 * <p>An iterator itself is not a tensor, so its shape and dtype are empty. However, it holds a
 * reference to the generator of the underlying iterable so that elements can be generated during
 * iteration (e.g., via {@code next()}).
 */
public class IteratorGenerator extends TensorGenerator implements DelegatingTensorGenerator {

  /** The generator for the underlying iterable. */
  private final TensorGenerator underlying;

  public IteratorGenerator(PointsToSetVariable source, TensorGenerator underlying) {
    super(source);
    this.underlying = underlying;
  }

  /**
   * Returns the generator for the underlying iterable.
   *
   * @return the underlying generator.
   */
  public TensorGenerator getUnderlying() {
    return underlying;
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    // An iterator is not a tensor (⊥).
    return HashSetFactory.make();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // An iterator is not a tensor (⊥).
    return Collections.emptySet();
  }

  @Override
  protected int getShapeParameterPosition() {
    return -1;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // An iterator is not a tensor (⊥).
    return Collections.emptySet();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return -1;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
