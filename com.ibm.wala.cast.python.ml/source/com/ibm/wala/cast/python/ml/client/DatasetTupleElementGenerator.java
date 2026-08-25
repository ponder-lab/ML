package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
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
  private final TupleElementProvider underlying;

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
      PointsToSetVariable source, TupleElementProvider underlying, int index) {
    super(source);
    this.underlying = underlying;
    this.index = index;
  }

  /**
   * Constructs a new {@code DatasetTupleElementGenerator} anchored on a call-graph node
   * (manual-generator path, used when no points-to source is available — e.g., producer delegation
   * for a tuple-element allocation inside a synthetic {@code do()} body; wala/ML#830).
   *
   * @param node the CG node the generator is anchored at
   * @param underlying the generator representing the underlying dataset
   * @param index the index of this element within the tuple
   */
  public DatasetTupleElementGenerator(CGNode node, TupleElementProvider underlying, int index) {
    super(node);
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
    return (TensorGenerator) underlying;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The tuple index: two positions of one batch tuple share the allocating {@code do()}
   *     node as their manual anchor, so class plus anchor alone would key both positions' memoized
   *     results (and the same-operation guard) identically, replaying one position's shapes for the
   *     other (wala/ML#830).
   */
  @Override
  protected Object operationDiscriminator() {
    return this.index;
  }

  @Override
  public String toString() {
    return "DatasetTupleElementGenerator(" + underlying + ", index=" + index + ")";
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation delegates to the underlying dataset generator for the specific
   *     tuple index.
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
   * @implNote This implementation delegates to the underlying dataset generator for the specific
   *     tuple index.
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
   * @implNote This implementation delegates to the underlying dataset generator for the specific
   *     tuple index.
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
   * @implNote This implementation returns {@code null} since properties are fully delegated and the
   *     shape cannot be determined here.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
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
   * @implNote This implementation returns {@link DType#UNKNOWN} since properties are fully
   *     delegated and dtype cannot be determined here.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
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
