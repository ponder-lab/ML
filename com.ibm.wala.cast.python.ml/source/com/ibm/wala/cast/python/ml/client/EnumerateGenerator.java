package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@link TensorType}s produced by Python's {@code enumerate} built-in function.
 *
 * <p>The {@code enumerate} function returns an iterator over tuples, where each tuple contains an
 * index and an element from the underlying iterable. This generator delegates shape and dtype
 * inference to the generator of the underlying iterable.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EnumerateGenerator extends TensorGenerator implements DelegatingTensorGenerator {

  /** The generator for the underlying iterable being enumerated. */
  private final TensorGenerator underlying;

  public EnumerateGenerator(PointsToSetVariable source, TensorGenerator underlying) {
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
    return underlying.getTensorTypes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return HashSetFactory.make();
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
    return HashSetFactory.make();
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
