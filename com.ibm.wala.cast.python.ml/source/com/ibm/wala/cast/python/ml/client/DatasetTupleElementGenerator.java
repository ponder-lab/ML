package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator representing a specific element within a tuple produced by a dataset. This is used
 * when a dataset yields structured elements (like tuples) and we need to track the shape and type
 * of individual components within that structure, delegating the lookup to the underlying generator
 * based on the tuple index.
 */
public class DatasetTupleElementGenerator extends TensorGenerator
    implements DelegatingTensorGenerator {

  /** The generator representing the underlying dataset this element belongs to. */
  private final DatasetFromGeneratorGenerator underlying;

  /** The index of this element within the tuple. */
  private final int index;

  /**
   * Constructs a new {@code DatasetTupleElementGenerator}.
   *
   * @param source the points-to set variable representing the source of the element
   * @param underlying the generator representing the underlying dataset
   * @param index the index of this element within the tuple
   */
  public DatasetTupleElementGenerator(
      PointsToSetVariable source, DatasetFromGeneratorGenerator underlying, int index) {
    super(source);
    this.underlying = underlying;
    this.index = index;
  }

  /**
   * Retrieves the underlying generator.
   *
   * @return the generator representing the underlying dataset
   */
  @Override
  public TensorGenerator getUnderlying() {
    return underlying;
  }

  @Override
  public String toString() {
    return "DatasetTupleElementGenerator(" + underlying + ", index=" + index + ")";
  }

  /**
   * {@inheritDoc}
   *
   * <p>This implementation delegates to the underlying dataset generator for the specific tuple
   * index.
   */
  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return underlying.getTensorTypesForIndex(builder, index);
    }
    return super.getTensorTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>This implementation delegates to the underlying dataset generator for the specific tuple
   * index.
   */
  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return underlying.getShapesForIndex(builder, index);
    }
    return super.getShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>This implementation delegates to the underlying dataset generator for the specific tuple
   * index.
   */
  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return underlying.getDTypesForIndex(builder, index);
    }
    return super.getDTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>This implementation returns an empty set since properties are fully delegated.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
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

  /**
   * {@inheritDoc}
   *
   * <p>This implementation returns an empty set since properties are fully delegated.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
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
