package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
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
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** A generator for datasets created by {@code tf.data.Dataset.zip}. */
public class DatasetZipGenerator extends DatasetGenerator implements TupleElementProvider {

  /** The dtype name of a component of unknown type. */
  private static final String UNKNOWN_DTYPE = DType.UNKNOWN.name().toLowerCase(Locale.ROOT);

  protected enum Parameters {
    DATASETS,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetZipGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetZipGenerator(CGNode node) {
    super(node);
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  /**
   * The generators of the datasets a selector picks from the zipped structure: an {@link Integer}
   * index into a tuple or list of datasets, or a {@link String} key into a dict of datasets.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param selector The index or the key.
   * @return The selected datasets' generators, possibly empty.
   */
  private List<TensorGenerator> getDatasetsFor(
      PropagationCallGraphBuilder builder, Object selector) {
    List<TensorGenerator> ret = new ArrayList<>();
    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());
    if (datasetsPTS == null) return ret;
    for (InstanceKey ik : datasetsPTS) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      TypeReference type = asin.concreteType().getReference();
      FieldReference field;
      if (selector instanceof Integer index && (type.equals(list) || type.equals(tuple)))
        field = FieldReference.findOrCreate(type, findOrCreateAsciiAtom(index.toString()), Root);
      else if (selector instanceof String key && type.equals(dict))
        field = FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(key), Root);
      else continue;
      IField f = builder.getClassHierarchy().resolveField(field);
      if (f == null) continue;
      PointerKey fieldPK = builder.getPointerKeyForInstanceField(asin, f);
      if (fieldPK == null || builder.getPropagationSystem().isImplicit(fieldPK)) continue;
      PointsToSetVariable var = builder.getPropagationSystem().findOrCreatePointsToSet(fieldPK);
      try {
        TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
        if (generator != null) ret.add(generator);
      } catch (IllegalArgumentException e) {
        // A field holding no dataset contributes nothing.
      }
    }
    return ret;
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());

    if (datasetsPTS != null && !datasetsPTS.isEmpty()) {
      Set<TensorType> ret = HashSetFactory.make();
      for (InstanceKey ik : datasetsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.concreteType().getReference().equals(list)
                || asin.concreteType().getReference().equals(tuple))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null) {
              Set<TensorType> types = this.getTensorTypesForIndex(builder, fieldIndex);
              // A component of unknown type makes the element's unknown.
              if (types == null) return null;
              ret.addAll(types);
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return super.getTensorTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union of the element types of the datasets at the index, unknown when any of them
   *     is.
   */
  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<TensorType> ret = HashSetFactory.make();
    for (TensorGenerator dataset : this.getDatasetsFor(builder, index)) {
      Set<TensorType> types = dataset.getTensorTypes(builder);
      // An unknown component is a tensor of unknown shape and dtype.
      if (types == null) return Set.of(new TensorType(UNKNOWN_DTYPE, null));
      ret.addAll(types);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union of the element shapes of the datasets at the index, unknown when any of
   *     them is.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : this.getDatasetsFor(builder, index)) {
      Set<List<Dimension<?>>> shapes = dataset.getShapes(builder);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union of the element dtypes of the datasets at the index.
   */
  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : this.getDatasetsFor(builder, index)) {
      Set<DType> dtypes = dataset.getDTypes(builder);
      if (dtypes != null) ret.addAll(dtypes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The first selector picks a zipped dataset, by index from a tuple or list of datasets
   *     or by key from a dict of datasets, and the rest of the path is that dataset's.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.getDatasetsFor(builder, path.get(0));
    if (datasets.isEmpty()) return false;
    List<Object> rest = path.subList(1, path.size());
    if (rest.isEmpty()) return true;
    for (TensorGenerator dataset : datasets)
      if (!(dataset instanceof TupleElementProvider tep && tep.resolvesPath(builder, rest)))
        return false;
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets the first selector picks of the shapes at the rest of the
   *     path, unknown when any of them is.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getShapesForIndex(builder, index);
    List<TensorGenerator> datasets = this.getDatasetsFor(builder, path.get(0));
    if (datasets.isEmpty()) return null;
    List<Object> rest = path.subList(1, path.size());
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : datasets) {
      Set<List<Dimension<?>>> shapes;
      if (rest.isEmpty()) shapes = dataset.getShapes(builder);
      else if (dataset instanceof TupleElementProvider tep)
        shapes = tep.getShapesForPath(builder, rest);
      else shapes = null;
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets the first selector picks of the dtypes at the rest of the
   *     path, unknown when any of them is.
   */
  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getDTypesForIndex(builder, index);
    List<TensorGenerator> datasets = this.getDatasetsFor(builder, path.get(0));
    if (datasets.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    List<Object> rest = path.subList(1, path.size());
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : datasets) {
      Set<DType> dtypes;
      if (rest.isEmpty()) dtypes = dataset.getDTypes(builder);
      else if (dataset instanceof TupleElementProvider tep)
        dtypes = tep.getDTypesForPath(builder, rest);
      else dtypes = null;
      if (dtypes == null || dtypes.isEmpty() || dtypes.contains(DType.UNKNOWN))
        return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }
}
