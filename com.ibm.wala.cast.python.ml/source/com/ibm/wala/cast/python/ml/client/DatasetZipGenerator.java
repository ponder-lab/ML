package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
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
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
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
      TensorGenerator generator = generatorOf(builder, fieldOf(builder, ik, selector));
      if (generator != null) ret.add(generator);
    }
    return ret;
  }

  /**
   * A zipped dataset a path reaches, with the rest of the path, which selects within that dataset's
   * element.
   *
   * @param dataset The dataset's generator.
   * @param rest The selectors below the dataset.
   */
  private record Zipped(TensorGenerator dataset, List<Object> rest) {}

  /**
   * What a path reaches in the structure of datasets {@code zip} is given.
   *
   * @param datasets The datasets the path reaches, each with the rest of the path.
   * @param structure Whether the path also ends within the structure, at a tuple, list or dict of
   *     datasets rather than at a dataset.
   */
  private record Reach(List<Zipped> datasets, boolean structure) {}

  /**
   * The zipped datasets a path reaches (wala/ML#1208): its selectors walk the structure of datasets
   * {@code zip} is given, which may nest, as {@code zip(((x, m), y))}'s does, until they reach
   * datasets, and the rest of the path is theirs.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param path The selectors, outermost first.
   * @return The datasets reached, and whether the path also ends within the structure.
   */
  private Reach reach(PropagationCallGraphBuilder builder, List<Object> path) {
    List<Zipped> datasets = new ArrayList<>();
    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());
    Set<InstanceKey> structures = HashSetFactory.make();
    if (datasetsPTS != null) for (InstanceKey ik : datasetsPTS) structures.add(ik);
    for (int i = 0; i < path.size() && !structures.isEmpty(); i++) {
      Set<InstanceKey> next = HashSetFactory.make();
      for (InstanceKey structure : structures) {
        PointerKey field = fieldOf(builder, structure, path.get(i));
        if (field == null) continue;
        boolean dataset = false;
        for (InstanceKey member : builder.getPointerAnalysis().getPointsToSet(field))
          if (isStructureOfDatasets(member)) next.add(member);
          else dataset = true;
        TensorGenerator generator = dataset ? generatorOf(builder, field) : null;
        if (generator != null)
          datasets.add(new Zipped(generator, path.subList(i + 1, path.size())));
      }
      structures = next;
    }
    return new Reach(datasets, !structures.isEmpty());
  }

  /**
   * Whether a key the zipped structure holds is itself a structure of datasets rather than a
   * dataset: one of the classes the element's copy takes for a structure, a tuple, a dict or a
   * namedtuple ({@link TensorFlowTypes#DATASET_ELEMENT_STRUCTURE_TYPES}), or a list.
   *
   * @param key The key.
   * @return {@code true} iff the key is such a structure.
   */
  private static boolean isStructureOfDatasets(InstanceKey key) {
    AllocationSiteInNode asin = getAllocationSiteInNode(key);
    if (asin == null) return false;
    TypeReference type = asin.concreteType().getReference();
    return type.equals(list) || TensorFlowTypes.DATASET_ELEMENT_STRUCTURE_TYPES.contains(type);
  }

  /**
   * The field of a structure of datasets a selector names: an {@link Integer} index into a tuple or
   * list, or a {@link String} key into a dict.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param structure The structure.
   * @param selector The index or the key.
   * @return The field's pointer key, or {@code null} when the selector names none of the structure.
   */
  private static PointerKey fieldOf(
      PropagationCallGraphBuilder builder, InstanceKey structure, Object selector) {
    AllocationSiteInNode asin = getAllocationSiteInNode(structure);
    if (asin == null) return null;
    TypeReference type = asin.concreteType().getReference();
    FieldReference field;
    if (selector instanceof Integer index && (type.equals(list) || type.equals(tuple)))
      field = FieldReference.findOrCreate(type, findOrCreateAsciiAtom(index.toString()), Root);
    else if (selector instanceof String key && type.equals(dict))
      field = FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(key), Root);
    // A namedtuple's members are its positions, as the element's copy keys them.
    else if (selector instanceof Integer index
        && type.equals(PythonSSAPropagationCallGraphBuilder.NAMED_TUPLE_INSTANCE))
      field = FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(index.toString()), Root);
    else return null;
    IField f = builder.getClassHierarchy().resolveField(field);
    if (f == null) return null;
    PointerKey ret = builder.getPointerKeyForInstanceField(asin, f);
    return ret == null || builder.getPropagationSystem().isImplicit(ret) ? null : ret;
  }

  /**
   * The generator of the datasets a field of the zipped structure holds.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param field The field, or {@code null}.
   * @return The generator, or {@code null} when the field holds no dataset.
   */
  private static TensorGenerator generatorOf(
      PropagationCallGraphBuilder builder, PointerKey field) {
    if (field == null) return null;
    try {
      return TensorGeneratorFactory.getGenerator(
          builder.getPropagationSystem().findOrCreatePointsToSet(field), builder);
    } catch (IllegalArgumentException e) {
      // A field holding no dataset contributes nothing.
      return null;
    }
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
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

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
   * @implNote The selectors walk the structure of datasets {@code zip} is given, by index into a
   *     tuple or list or by key into a dict, until they reach a dataset, and the rest of the path
   *     is that dataset's.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    Reach reach = this.reach(builder, path);
    if (reach.datasets().isEmpty()) return reach.structure();
    for (Zipped zipped : reach.datasets())
      if (!zipped.rest().isEmpty()
          && !(zipped.dataset() instanceof TupleElementProvider tep
              && tep.resolvesPath(builder, zipped.rest()))) return false;
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote With no selectors, the component is the whole element, the structure of datasets
   *     {@code zip} is given, as an enumeration of the zip asks. The selectors walk that structure,
   *     which may nest, until they reach zipped datasets: a path ending within it is a structure;
   *     with no more selectors, the component is a reached dataset's whole element, a structure
   *     when every reached dataset yields a tuple, as {@code from_tensor_slices((x, mask))} does;
   *     otherwise the rest of the path is every reached dataset's.
   */
  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.isEmpty()) return this.yieldsTuple(builder);
    Reach reach = this.reach(builder, path);
    if (reach.datasets().isEmpty()) return reach.structure();
    for (Zipped zipped : reach.datasets()) {
      if (!(zipped.dataset() instanceof TupleElementProvider tep)) return false;
      if (zipped.rest().isEmpty()
          ? !tep.yieldsTuple(builder)
          : !tep.isStructureAt(builder, zipped.rest())) return false;
    }
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The selectors walk the zipped structure to datasets, and the rest of the path is
   *     every reached dataset's, whose declared lengths must agree. The zipped tuple itself, the
   *     empty path, and a structure of datasets within it are not reported.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.isEmpty()) return OptionalInt.empty();
    Reach reach = this.reach(builder, path);
    if (reach.datasets().isEmpty() || reach.structure()) return OptionalInt.empty();
    int ret = -1;
    for (Zipped zipped : reach.datasets()) {
      if (!(zipped.dataset() instanceof TupleElementProvider tep)) return OptionalInt.empty();
      OptionalInt arity = tep.declaredArity(builder, zipped.rest());
      if (arity.isEmpty() || (ret >= 0 && ret != arity.getAsInt())) return OptionalInt.empty();
      ret = arity.getAsInt();
    }
    return OptionalInt.of(ret);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets the selectors reach of the shapes at the rest of the
   *     path, unknown when any of them is.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getShapesForIndex(builder, index);
    Reach reach = this.reach(builder, path);
    // A structure of datasets within the zipped structure has no shape of its own.
    if (reach.datasets().isEmpty() || reach.structure()) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Zipped zipped : reach.datasets()) {
      TensorGenerator dataset = zipped.dataset();
      List<Object> rest = zipped.rest();
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
   * @implNote The union over the datasets the selectors reach of the dtypes at the rest of the
   *     path, unknown when any of them is.
   */
  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getDTypesForIndex(builder, index);
    Reach reach = this.reach(builder, path);
    if (reach.datasets().isEmpty() || reach.structure()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Zipped zipped : reach.datasets()) {
      TensorGenerator dataset = zipped.dataset();
      List<Object> rest = zipped.rest();
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

  /** How deep {@link #sizesOf} follows a nested structure of datasets. */
  private static final int SIZE_DEPTH = 8;

  /**
   * {@inheritDoc}
   *
   * @implNote A zip has as many elements as its shortest dataset (wala/ML#1203), so each size it
   *     may have is the least of one size of each dataset it zips. The sizes are unknown when any
   *     zipped dataset's are, or when the structure holds anything but datasets and structures of
   *     them.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());
    if (datasetsPTS == null || datasetsPTS.isEmpty()) return Collections.emptySet();
    Set<Long> ret = this.sizesOf(builder, datasetsPTS, 0);
    return ret == null ? Collections.emptySet() : ret;
  }

  /**
   * The sizes a structure of datasets, or a dataset, may have when zipped. The members of a
   * points-to set are alternatives, so their sizes join; the fields of one structure are zipped
   * together, so a size is the least of one size of each.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param members The keys the zipped value may be.
   * @param depth How many structures the walk has entered.
   * @return The sizes, or {@code null} when some member's size is unknown.
   */
  private Set<Long> sizesOf(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> members, int depth) {
    Set<Long> ret = HashSetFactory.make();
    for (InstanceKey member : members) {
      // The structures are those the path reads walk (wala/ML#1208), a namedtuple among them.
      if (!isStructureOfDatasets(member) || depth >= SIZE_DEPTH) return null;
      Set<Long> sizes = this.structureSizes(builder, member, depth);
      if (sizes == null) return null;
      ret.addAll(sizes);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * The sizes a structure of datasets may have when zipped: the least of one size of each field.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param structure The structure, a tuple, list, dict or namedtuple.
   * @param depth How many structures the walk has entered.
   * @return The sizes, or {@code null} when some field's size is unknown.
   */
  private Set<Long> structureSizes(
      PropagationCallGraphBuilder builder, InstanceKey structure, int depth) {
    OrdinalSet<InstanceKey> catalog =
        withoutClientOnlyKeys(
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                        .getPointerKeyForObjectCatalog(structure)));
    boolean keyed = structure.concreteType().getReference().equals(dict);
    Set<Long> ret = null;
    for (InstanceKey name : catalog) {
      if (!(name instanceof ConstantKey<?> constant)) return null;
      // A dict's fields are named by its keys, a positional structure's by its positions, as the
      // path reads select them.
      Integer index = getFieldIndex(constant);
      Object selector = keyed ? constant.getValue() : index;
      PointerKey fieldPK = selector == null ? null : fieldOf(builder, structure, selector);
      if (fieldPK == null) return null;
      Set<Long> sizes = this.fieldSizes(builder, fieldPK, depth);
      if (sizes == null) return null;
      if (ret == null) ret = sizes;
      else {
        Set<Long> least = HashSetFactory.make();
        for (Long a : ret) for (Long b : sizes) least.add(Math.min(a, b));
        ret = least;
      }
    }
    return ret;
  }

  /**
   * The sizes the datasets, or structures of them, a field of the zipped structure holds may have.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param fieldPK The field.
   * @param depth How many structures the walk has entered.
   * @return The sizes, or {@code null} when they are unknown.
   */
  private Set<Long> fieldSizes(PropagationCallGraphBuilder builder, PointerKey fieldPK, int depth) {
    OrdinalSet<InstanceKey> members =
        withoutClientOnlyKeys(builder.getPointerAnalysis().getPointsToSet(fieldPK));
    if (members.isEmpty()) return null;
    boolean structures = false;
    boolean datasets = false;
    for (InstanceKey member : members)
      if (isStructureOfDatasets(member)) structures = true;
      else datasets = true;
    // A field that may be a dataset or a structure of datasets is left unknown.
    if (structures && datasets) return null;
    if (structures) return this.sizesOf(builder, members, depth + 1);
    TensorGenerator generator;
    try {
      generator =
          TensorGeneratorFactory.getGenerator(
              builder.getPropagationSystem().findOrCreatePointsToSet(fieldPK), builder);
    } catch (IllegalArgumentException e) {
      return null;
    }
    if (!(generator instanceof DatasetGenerator dataset)) return null;
    Set<Long> sizes = dataset.getDatasetSizes(builder);
    return sizes == null || sizes.isEmpty() ? null : sizes;
  }
}
