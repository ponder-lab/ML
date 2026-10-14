package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * The join of the generators of the datasets an iterable may be, such as an attribute several
 * writes store a dataset to, kept structured: a component of the element, selected by an index or a
 * path of keys, is the join of the parts' components. A part that cannot answer for a component
 * makes the component unknown.
 */
public class DatasetCreatorJoin extends CreatorJoin implements TupleElementProvider {

  /**
   * Constructs the join of the given dataset generators, anchored at the value they may be.
   *
   * @param source The value's {@link PointsToSetVariable}.
   * @param parts The datasets' generators, at least two.
   */
  DatasetCreatorJoin(PointsToSetVariable source, List<TensorGenerator> parts) {
    super(source, parts);
  }

  /**
   * The parts as element providers.
   *
   * @return The parts, or {@code null} when some part provides no element structure.
   */
  private List<TupleElementProvider> providers() {
    List<TupleElementProvider> providers = new java.util.ArrayList<>();
    for (TensorGenerator part : this.getParts())
      if (part instanceof TupleElementProvider provider) providers.add(provider);
      else return null;
    return providers;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    List<TupleElementProvider> providers = this.providers();
    if (providers == null) return false;
    for (TupleElementProvider provider : providers)
      if (!provider.yieldsTuple(builder)) return false;
    return true;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    List<TupleElementProvider> providers = this.providers();
    if (providers == null) return null;
    Set<TensorType> joined = HashSetFactory.make();
    for (TupleElementProvider provider : providers) {
      Set<TensorType> types = provider.getTensorTypesForIndex(builder, index);
      if (types == null) return null;
      joined.addAll(types);
    }
    return joined;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The shapes of the joined component's types.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<TensorType> types = this.getTensorTypesForIndex(builder, index);
    if (types == null) return null;
    Set<List<Dimension<?>>> shapes = HashSetFactory.make();
    for (TensorType type : types) {
      if (type.getDims() == null) return null;
      shapes.add(type.getDims());
    }
    return shapes;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The dtypes of the joined component's types.
   */
  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<TensorType> types = this.getTensorTypesForIndex(builder, index);
    if (types == null) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = EnumSet.noneOf(DType.class);
    for (TensorType type : types) dtypes.add(type.getDType());
    return dtypes;
  }

  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TupleElementProvider> providers = this.providers();
    if (providers == null) return false;
    for (TupleElementProvider provider : providers)
      if (!provider.resolvesPath(builder, path)) return false;
    return true;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    List<TupleElementProvider> providers = this.providers();
    if (providers == null) return null;
    Set<List<Dimension<?>>> joined = HashSetFactory.make();
    for (TupleElementProvider provider : providers) {
      Set<List<Dimension<?>>> shapes = provider.getShapesForPath(builder, path);
      if (shapes == null) return null;
      joined.addAll(shapes);
    }
    return joined;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TupleElementProvider> providers = this.providers();
    if (providers == null) return EnumSet.of(DType.UNKNOWN);
    Set<DType> joined = EnumSet.noneOf(DType.class);
    for (TupleElementProvider provider : providers)
      joined.addAll(provider.getDTypesForPath(builder, path));
    return joined;
  }
}
