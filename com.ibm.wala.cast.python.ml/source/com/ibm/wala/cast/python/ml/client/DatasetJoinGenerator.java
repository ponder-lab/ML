package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;

/**
 * A generator for a dataset whose element is any one of the elements of several datasets, as {@code
 * concatenate}, {@code sample_from_datasets} and {@code choose_from_datasets} make (wala/ML#1215).
 * The datasets are its {@link #getReceiverGenerators receivers}, and a read of the element joins
 * over them, so a tuple element stays a tuple whose members keep their own types.
 */
public abstract class DatasetJoinGenerator extends DatasetGenerator {

  protected DatasetJoinGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetJoinGenerator(CGNode node) {
    super(node);
  }

  /**
   * The generators of the datasets a sequence holds, a list or a tuple passed as the datasets to
   * join.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param sequences The keys the sequence may be.
   * @return The datasets' generators, without duplicates, or {@code null} when the sequence has no
   *     key, a key is no list or tuple, or a member does not resolve.
   */
  protected static List<TensorGenerator> datasetsIn(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> sequences) {
    if (sequences == null || sequences.isEmpty()) return null;
    List<TensorGenerator> ret = new ArrayList<>();
    for (InstanceKey sequence : sequences) {
      AllocationSiteInNode asin = getAllocationSiteInNode(sequence);
      if (asin == null) return null;
      TypeReference type = asin.concreteType().getReference();
      if (!type.equals(list) && !type.equals(tuple)) return null;
      OrdinalSet<InstanceKey> catalog =
          withoutClientOnlyKeys(
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin)));
      for (InstanceKey name : catalog) {
        if (!(name instanceof ConstantKey<?> constant)) continue;
        Integer index = getFieldIndex(constant);
        if (index == null) continue;
        IField f =
            builder
                .getClassHierarchy()
                .resolveField(
                    FieldReference.findOrCreate(
                        type, findOrCreateAsciiAtom(index.toString()), Root));
        if (f == null) return null;
        OrdinalSet<InstanceKey> members =
            withoutClientOnlyKeys(
                builder
                    .getPointerAnalysis()
                    .getPointsToSet(builder.getPointerKeyForInstanceField(asin, f)));
        if (members.isEmpty()) return null;
        for (InstanceKey member : members) {
          TensorGenerator generator = receiverGenerator(builder, member);
          if (generator == null) return null;
          if (!ret.contains(generator)) ret.add(generator);
        }
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The element is one of the datasets' elements, unchanged.
   */
  @Override
  public boolean elementFollowsReceiver() {
    return true;
  }

  /**
   * The datasets whose elements this one's is, or {@code null} for a view fixed to one of them,
   * which reads that one alone.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The datasets' generators, or {@code null}.
   */
  protected List<TensorGenerator> joined(PropagationCallGraphBuilder builder) {
    return this.getFixedReceiver() != null ? null : this.getReceiverGenerators(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDefaultShapes(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : datasets) {
      Set<List<Dimension<?>>> shapes = dataset.getShapes(builder);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDefaultDTypes(builder);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : datasets) {
      Set<DType> dtypes = dataset.getDTypes(builder);
      if (dtypes == null || dtypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getShapesForPath(builder, List.of(index));
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getDTypesForPath(builder, List.of(index));
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getTensorTypesForPath(builder, List.of(index));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A path resolves where some dataset whose element this one's is resolves it.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.resolvesPath(builder, path);
    for (TensorGenerator dataset : datasets)
      if (dataset instanceof TupleElementProvider tep && tep.resolvesPath(builder, path))
        return true;
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A length is declared only where each dataset declares the same one.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.declaredArity(builder, path);
    int ret = -1;
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof TupleElementProvider tep)) return OptionalInt.empty();
      OptionalInt arity = tep.declaredArity(builder, path);
      if (arity.isEmpty() || (ret >= 0 && ret != arity.getAsInt())) return OptionalInt.empty();
      ret = arity.getAsInt();
    }
    return ret < 0 ? OptionalInt.empty() : OptionalInt.of(ret);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets of their shapes at the path, unknown when any is.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getShapesForPath(builder, path);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof TupleElementProvider tep)) return null;
      Set<List<Dimension<?>>> shapes = tep.getShapesForPath(builder, path);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets of their dtypes at the path, unknown when any is.
   */
  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDTypesForPath(builder, path);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof TupleElementProvider tep)) return EnumSet.of(DType.UNKNOWN);
      Set<DType> dtypes = tep.getDTypesForPath(builder, path);
      if (dtypes == null || dtypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Each dataset's types at the path, so a shape stays paired with its own dataset's
   *     dtype, and the joined shapes and dtypes crossed when a dataset provides no path read.
   */
  @Override
  public Set<TensorType> getTensorTypesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    Set<TensorType> ret = HashSetFactory.make();
    if (datasets != null && datasets.stream().allMatch(TupleElementProvider.class::isInstance)) {
      for (TensorGenerator dataset : datasets)
        ret.addAll(((TupleElementProvider) dataset).getTensorTypesForPath(builder, path));
      return ret;
    }
    Set<List<Dimension<?>>> shapes = this.getShapesForPath(builder, path);
    for (DType dtype : this.getDTypesForPath(builder, path)) {
      String cellType = dtype.name().toLowerCase(Locale.ROOT);
      if (shapes == null) ret.add(new TensorType(cellType, null));
      else for (List<Dimension<?>> shape : shapes) ret.add(new TensorType(cellType, shape));
    }
    return ret;
  }

  /**
   * The sizes a dataset holding every element of each of the given datasets may have: each is the
   * sum of one size of each dataset.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param datasets The datasets.
   * @return The sizes, or an empty set, unknown, when a dataset's size is.
   */
  protected static Set<Long> sumOfSizes(
      PropagationCallGraphBuilder builder, List<TensorGenerator> datasets) {
    Set<Long> ret = null;
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof DatasetGenerator generator)) return Collections.emptySet();
      Set<Long> sizes = generator.getDatasetSizes(builder);
      if (sizes == null || sizes.isEmpty()) return Collections.emptySet();
      if (ret == null) ret = sizes;
      else {
        Set<Long> sums = HashSetFactory.make();
        for (Long a : ret) for (Long b : sizes) sums.add(a + b);
        ret = sums;
      }
    }
    return ret == null ? Collections.emptySet() : ret;
  }
}
