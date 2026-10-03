package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * The element a dataset iterator's {@code __next__} allocates (<a
 * href="https://github.com/wala/ML/issues/1010">wala/ML#1010</a>): the value a loop over a dataset
 * binds. It is typed from the generators of the datasets the element allocations came from, read
 * off the {@code dataset} field the iterator's summary writes on each element, joined over every
 * element allocation the typed value holds: a function entered from several datasets binds one
 * element per dataset, and reading only the first would type the value as one dataset's. It answers
 * the element's structure as a {@link TupleElementProvider} by delegation, so a component read
 * ({@code batch[0]}, {@code record["ids"]}, a nested path) dispatches through the factory's
 * existing element-path machinery to a {@link DatasetTupleElementGenerator} over the same dataset
 * generators.
 */
public class DatasetIteratorElementGenerator extends TensorGenerator
    implements TupleElementProvider, DelegatingTensorGenerator {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetIteratorElementGenerator.class.getName());

  /** The field the iterator's summary writes the dataset into on the element and its components. */
  static final FieldReference DATASET_FIELD =
      FieldReference.findOrCreate(
          PythonTypes.Root, Atom.findOrCreateUnicodeAtom("dataset"), PythonTypes.Root);

  /** The datasets' generators, one per element allocation that resolved; never {@code null}. */
  private final List<TensorGenerator> datasets;

  /**
   * A generator for an element read as a value.
   *
   * @param source The value's points-to set variable.
   * @param datasets The generators of the datasets the value's element allocations came from.
   */
  public DatasetIteratorElementGenerator(
      PointsToSetVariable source, List<TensorGenerator> datasets) {
    super(source);
    this.datasets = List.copyOf(datasets);
  }

  /**
   * A generator anchored on an element allocation inside the iterator's {@code __next__} node, for
   * a read of the allocation itself (a layer applied to the loop variable).
   *
   * @param node The allocating node.
   * @param datasets The generators of the datasets the allocation came from.
   */
  public DatasetIteratorElementGenerator(CGNode node, List<TensorGenerator> datasets) {
    super(node);
    this.datasets = List.copyOf(datasets);
  }

  /**
   * The element allocations among a value's points-to set.
   *
   * @param source The value's points-to set variable.
   * @param builder The propagation call graph builder.
   * @return The element allocations, possibly empty.
   */
  static List<AllocationSiteInNode> elementAllocations(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    List<AllocationSiteInNode> ret = new ArrayList<>();
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(source.getPointerKey())) {
      AllocationSiteInNode asin;
      try {
        asin = getAllocationSiteInNode(ik);
      } catch (IllegalArgumentException e) {
        continue;
      }
      if (asin != null
          && asin.concreteType().getReference().equals(TensorFlowTypes.DATASET_ELEMENT_TYPE))
        ret.add(asin);
    }
    return ret;
  }

  /**
   * The dataset allocations an element or component allocation's {@code dataset} field holds.
   *
   * @param allocation The element or component allocation.
   * @param builder The propagation call graph builder.
   * @return The dataset allocations, possibly empty.
   */
  static List<AllocationSiteInNode> datasetAllocationsOf(
      AllocationSiteInNode allocation, PropagationCallGraphBuilder builder) {
    List<AllocationSiteInNode> ret = new ArrayList<>();
    IField field = builder.getClassHierarchy().resolveField(DATASET_FIELD);
    if (field == null) return ret;
    OrdinalSet<InstanceKey> held =
        builder
            .getPointerAnalysis()
            .getPointsToSet(builder.getPointerKeyForInstanceField(allocation, field));
    if (held == null) return ret;
    for (InstanceKey ik : held) {
      AllocationSiteInNode dataset = getAllocationSiteInNode(ik);
      if (dataset != null) ret.add(dataset);
    }
    return ret;
  }

  /**
   * The generators of the datasets some element or component allocations came from, read off each
   * allocation's {@code dataset} field; an allocation whose field holds no dataset with a generator
   * contributes nothing.
   *
   * @param allocations The element or component allocations.
   * @param builder The propagation call graph builder.
   * @return The dataset generators, in allocation order, without duplicates.
   */
  static List<TensorGenerator> datasetsOf(
      Collection<AllocationSiteInNode> allocations, PropagationCallGraphBuilder builder) {
    // The field is declared on the Python root class, as every summary-written field is.
    IField field = builder.getClassHierarchy().resolveField(DATASET_FIELD);
    List<TensorGenerator> ret = new ArrayList<>();
    if (field == null) return ret;
    for (AllocationSiteInNode allocation : allocations) {
      OrdinalSet<InstanceKey> held =
          builder
              .getPointerAnalysis()
              .getPointsToSet(builder.getPointerKeyForInstanceField(allocation, field));
      if (held == null) continue;
      for (InstanceKey ik : held) {
        AllocationSiteInNode dataset = getAllocationSiteInNode(ik);
        if (dataset == null) continue;
        TensorGenerator generator = datasetGenerator(dataset, builder);
        if (generator != null && !ret.contains(generator)) ret.add(generator);
      }
    }
    LOGGER.fine(() -> "Dataset element's dataset generators: " + ret + ".");
    return ret;
  }

  private static TensorGenerator datasetGenerator(
      AllocationSiteInNode dataset, PropagationCallGraphBuilder builder) {
    int vn = findDefinition(dataset.getNode(), dataset);
    if (vn > 0) {
      PointerKey pk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(dataset.getNode(), vn);
      if (!builder.getPropagationSystem().isImplicit(pk)) {
        TensorGenerator generator =
            TensorGeneratorFactory.getGenerator(
                builder.getPropagationSystem().findOrCreatePointsToSet(pk), builder);
        if (generator != null) return generator;
      }
    }
    return createManualGenerator(dataset.getNode(), dataset, builder);
  }

  @Override
  public TensorGenerator getUnderlying() {
    // One dataset is the delegation the factory's element-path dispatch unwraps; several are
    // joined here and unwrap to none.
    return this.datasets.size() == 1 ? this.datasets.get(0) : null;
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    if (this.datasets.isEmpty()) return super.getTensorTypes(builder);
    Set<TensorType> ret = HashSetFactory.make();
    for (TensorGenerator dataset : this.datasets) ret.addAll(dataset.getTensorTypes(builder));
    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    if (this.datasets.isEmpty()) return super.getShapes(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : this.datasets) {
      // Diverted through the engine's memo layer (wala/ML#365), as the element read is.
      Set<List<Dimension<?>>> shapes = memoizedShapeResult(builder, dataset).toLegacy();
      if (shapes == null) return null; // A dataset of unknown element shape makes the join unknown.
      ret.addAll(shapes);
    }
    return ret;
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    if (this.datasets.isEmpty()) return super.getDTypes(builder);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : this.datasets) ret.addAll(memoizedDTypes(builder, dataset));
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
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
  protected int getDTypeParameterPosition() {
    return -1;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  private List<TupleElementProvider> providers() {
    List<TupleElementProvider> ret = new ArrayList<>();
    for (TensorGenerator dataset : this.datasets)
      if (dataset instanceof TupleElementProvider tep) ret.add(tep);
    return ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    for (TupleElementProvider tep : providers()) if (tep.yieldsTuple(builder)) return true;
    return false;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<TensorType> ret = HashSetFactory.make();
    for (TupleElementProvider tep : providers())
      ret.addAll(tep.getTensorTypesForIndex(builder, index));
    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TupleElementProvider tep : providers()) {
      Set<List<Dimension<?>>> shapes = tep.getShapesForIndex(builder, index);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TupleElementProvider tep : providers()) ret.addAll(tep.getDTypesForIndex(builder, index));
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    for (TupleElementProvider tep : providers()) if (tep.resolvesPath(builder, path)) return true;
    return false;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TupleElementProvider tep : providers()) {
      Set<List<Dimension<?>>> shapes = tep.getShapesForPath(builder, path);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TupleElementProvider tep : providers()) ret.addAll(tep.getDTypesForPath(builder, path));
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  @Override
  public Set<TensorType> getTensorTypesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    Set<TensorType> ret = HashSetFactory.make();
    for (TupleElementProvider tep : providers())
      ret.addAll(tep.getTensorTypesForPath(builder, path));
    return ret;
  }

  @Override
  public String toString() {
    return "DatasetIteratorElementGenerator(" + this.datasets + ")";
  }
}
