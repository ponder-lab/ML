package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATA_PACKAGE_PREFIX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIT_DATA_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNPACKED_DATA_TYPE;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutPythonScalarKeys;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
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

/**
 * The data a Keras {@code fit}, {@code evaluate} or {@code predict} summary packs for the model's
 * step, and the result of {@code tf.keras.utils.unpack_x_y_sample_weight} (wala/ML#997).
 *
 * <p>Neither is a tensor; each is a tuple whose components the step reads by constant index ({@code
 * x, y = data}), so this generator is a {@link TupleElementProvider} and resolves no type of its
 * own. The pointer analysis already stores the packed values in the allocation's numeric slots and
 * the reads find them there; what the slots cannot carry is a tensor type, because the dataflow
 * seeds invoke results and parameters, not heap stores, so a read of slot {@code i} is typed here
 * from the slot's own values, the way a structured batch tuple is (wala/ML#830).
 *
 * <p>A packed {@code fit(x, y)} has {@code x} in slot 0 and {@code y} in slot 1, and index {@code
 * i} is slot {@code i}'s type, except when {@code x} is a {@code tf.data.Dataset}: Keras then
 * iterates it and the step receives an element, so index {@code i} is the element's component
 * {@code i} when the element is a tuple and the whole element at index 0 otherwise. An unpack
 * result has {@code data} in slot 0, and index {@code i} is {@code data}'s component {@code i} when
 * {@code data} is a tuple (a pack, a dataset element, or a tuple literal) and {@code data} itself
 * at index 0 otherwise, which is what {@code unpack_x_y_sample_weight} returns.
 *
 * <p>An element's component that is itself a structure, as {@code (x, mask)} in a {@code ((x,
 * mask), y)} element ({@link TupleElementProvider#isStructureAt}), is no tensor (⊥): the step hands
 * the model that tuple or dict, whose members are read at longer paths.
 */
public class FitDataGenerator extends TensorGenerator implements TupleElementProvider {

  /** The allocation whose slots hold the packed values. */
  private final AllocationSiteInNode allocation;

  /** Whether this is an unpack result ({@code true}) or a pack ({@code false}). */
  private final boolean unpack;

  /**
   * Constructs the generator for a pack or unpack allocation.
   *
   * @param node The summary node that allocated the data.
   * @param allocation The allocation.
   */
  public FitDataGenerator(CGNode node, AllocationSiteInNode allocation) {
    super(node);
    this.allocation = allocation;
    this.unpack = UNPACKED_DATA_TYPE.equals(allocation.concreteType().getReference());
  }

  /**
   * Whether this generator describes allocations of the given type.
   *
   * @param type An allocation's type.
   * @return {@code true} iff the type is a pack or an unpack result.
   */
  public static boolean describes(TypeReference type) {
    return FIT_DATA_TYPE.equals(type) || UNPACKED_DATA_TYPE.equals(type);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet(); // The data is a tuple, not a tensor (⊥).
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.noneOf(DType.class);
  }

  /** The data has no shape argument; its components are typed from the slots. */
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

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    boolean unknown = false;
    for (Component component : this.components(builder, List.of(index))) {
      Set<List<Dimension<?>>> shapes = component.shapes(builder);
      if (shapes == null) unknown = true;
      else ret.addAll(shapes);
    }
    return unknown && ret.isEmpty() ? null : ret;
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<DType> ret = this.tensorDTypesForIndex(builder, index);
    // A component that may also be a structure has the tensors' dtypes as a partial union, which
    // the UNKNOWN marker records (wala/ML#862), so a dtype reader takes none of them as certain.
    if (this.mayBeStructure(builder, index)) ret.add(DType.UNKNOWN);
    return ret;
  }

  /**
   * The dtypes of the tensors a component read at an index may be, without the marker of a
   * component that may also be a structure.
   *
   * @param builder The builder.
   * @param index The index.
   * @return The dtypes.
   */
  private Set<DType> tensorDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Component component : this.components(builder, List.of(index))) {
      Set<DType> dtypes = component.dtypes(builder);
      if (dtypes != null) ret.addAll(dtypes);
    }
    return ret;
  }

  /**
   * Whether a component read at an index is a tensor on some of the datasets the data may come from
   * and a structure on others.
   *
   * @param builder The builder.
   * @param index The index.
   * @return {@code true} iff the component may be either.
   */
  private boolean mayBeStructure(PropagationCallGraphBuilder builder, int index) {
    List<Component> components = this.components(builder, List.of(index));
    if (components == null) return false;
    for (Component component : components) if (component.mayBeStructure(builder)) return true;
    return false;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dtypes = this.tensorDTypesForIndex(builder, index);
    Set<TensorType> ret = HashSetFactory.make();
    if (shapes == null) {
      for (DType dtype : dtypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), null));
      return ret;
    }
    for (List<Dimension<?>> shape : shapes)
      for (DType dtype : dtypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), shape));
    // A component that is a structure on some of its datasets and a tensor on others is, at run
    // time, either; a wholly unknown member beside the tensors keeps a reader from taking the
    // tensors for all of it.
    if (this.mayBeStructure(builder, index))
      ret.add(new TensorType(DType.UNKNOWN.name().toLowerCase(Locale.ROOT), null));
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A longer path selects within component {@code path[0]}, which this data reaches only
   *     through a dataset's element or another pack: the step reads {@code inputs, targets = data},
   *     and a model's {@code call} then reads within {@code inputs}, as in {@code x, mask =
   *     inputs}.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1) return TupleElementProvider.super.resolvesPath(builder, path);
    List<Component> components = this.components(builder, path);
    return components != null && !components.isEmpty();
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getShapesForIndex(builder, index);
    List<Component> components = this.components(builder, path);
    if (components == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    boolean unknown = false;
    for (Component component : components) {
      Set<List<Dimension<?>>> shapes = component.shapes(builder);
      if (shapes == null) unknown = true;
      else ret.addAll(shapes);
    }
    return unknown && ret.isEmpty() ? null : ret;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getDTypesForIndex(builder, index);
    List<Component> components = this.components(builder, path);
    if (components == null) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Component component : components) {
      Set<DType> dtypes = component.dtypes(builder);
      if (dtypes != null) ret.addAll(dtypes);
    }
    return ret;
  }

  /**
   * The values a read of the given path resolves to, each with the way it is read.
   *
   * @param builder The builder.
   * @param path The selectors, outermost first; never empty.
   * @return The components, or {@code null} when the path is longer than one selector and a member
   *     of the data cannot be followed along it: only a dataset's element and another pack can. One
   *     such member makes the whole read unknown, though the others resolve, where a single index
   *     keeps the members that resolve.
   */
  private List<Component> components(PropagationCallGraphBuilder builder, List<Object> path) {
    if (!(path.get(0) instanceof Integer index)) return path.size() == 1 ? List.of() : null;
    boolean nested = path.size() > 1;
    List<Component> ret = new ArrayList<>();
    if (this.unpack) {
      // A dataset element read in a loop is the dataset object in the pointer analysis, whose set
      // also carries the object's field values (its method objects, its source tensors); the
      // dataset reading covers the element, and those members are not components of it. A tensor
      // unpacked at the same site as such an element would be read as that element too, an
      // imprecision accepted here, since the field values cannot be told from it by type.
      for (InstanceKey data : this.slot(builder, 0)) {
        AllocationSiteInNode asin = getAllocationSiteInNode(data);
        if (asin != null && isDataset(asin.concreteType().getReference())) {
          Component element =
              this.ofDataset(builder, this.slotVariable(builder, this.allocation, 0), path);
          return element == null ? null : List.of(element);
        }
      }
      for (InstanceKey data : this.slot(builder, 0)) {
        AllocationSiteInNode asin = getAllocationSiteInNode(data);
        TypeReference type = asin == null ? null : asin.concreteType().getReference();
        if (asin != null && describes(type)) {
          FitDataGenerator pack = new FitDataGenerator(asin.getNode(), asin);
          if (nested && !pack.resolvesPath(builder, path)) return null;
          ret.add(this.ofProvider(pack, path));
        } else if (nested) return null;
        else if (asin != null && PythonTypes.tuple.equals(type))
          ret.add(this.ofValues(this.field(builder, asin, index)));
        else if (index == 0) ret.add(this.ofValues(this.singleton(builder, data)));
      }
      return ret;
    }
    // A pack's slot holds the user's `x` as passed, so its members are the callers' own values: a
    // dataset member reads as the element's component through the slot's variable, and every other
    // member contributes its own type at index 0, so a step fed a tensor at one call and a dataset
    // at another, through one site, sees both.
    boolean dataset = false;
    for (InstanceKey x : this.slot(builder, 0)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(x);
      if (asin != null && isDataset(asin.concreteType().getReference())) {
        if (!dataset) {
          Component element =
              this.ofDataset(builder, this.slotVariable(builder, this.allocation, 0), path);
          if (element == null) return null;
          ret.add(element);
        }
        dataset = true;
      } else if (nested) return null;
      else if (index == 0) ret.add(this.ofValues(this.singleton(builder, x)));
    }
    if (nested) return ret;
    if (index > 0) {
      // The targets and weights of a tensor-fed call are the pack's own later slots; a dataset-fed
      // call's are the element's components, read above.
      OrdinalSet<InstanceKey> slot = this.slot(builder, index);
      if (!slot.isEmpty()) ret.add(this.ofValues(slot));
    }
    return ret;
  }

  /** The points-to variable of an allocation's numeric slot, or {@code null} if it has none. */
  private PointsToSetVariable slotVariable(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin, int index) {
    IField f = this.slotField(builder, index);
    if (f == null) return null;
    PointerKey key = builder.getPointerKeyForInstanceField(asin, f);
    if (builder.getPropagationSystem().isImplicit(key)) return null;
    return builder.getPropagationSystem().findOrCreatePointsToSet(key);
  }

  /** The {@code Root} field of a numeric slot, or {@code null} if the hierarchy has none. */
  private IField slotField(PropagationCallGraphBuilder builder, int index) {
    return builder
        .getClassHierarchy()
        .resolveField(
            FieldReference.findOrCreate(
                PythonTypes.Root,
                findOrCreateAsciiAtom(Integer.toString(index)),
                PythonTypes.Root));
  }

  /** The points-to set of this allocation's numeric slot. */
  private OrdinalSet<InstanceKey> slot(PropagationCallGraphBuilder builder, int index) {
    return this.field(builder, this.allocation, index);
  }

  /** The points-to set of an allocation's numeric field, empty when the field does not resolve. */
  private OrdinalSet<InstanceKey> field(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin, int index) {
    IField f = this.slotField(builder, index);
    if (f == null) return OrdinalSet.empty();
    OrdinalSet<InstanceKey> pts =
        withoutPythonScalarKeys(
            builder
                .getPointerAnalysis()
                .getPointsToSet(builder.getPointerKeyForInstanceField(asin, f)));
    return pts == null ? OrdinalSet.empty() : pts;
  }

  /** A one-member points-to set over the builder's instance key numbering. */
  private OrdinalSet<InstanceKey> singleton(PropagationCallGraphBuilder builder, InstanceKey key) {
    return OrdinalSet.toOrdinalSet(
        Collections.singleton(key), builder.getPointerAnalysis().getInstanceKeyMapping());
  }

  private static boolean isDataset(TypeReference type) {
    return type != null && type.getName().toString().startsWith(DATA_PACKAGE_PREFIX);
  }

  /** One way a component is read. */
  private interface Component {
    Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder builder);

    Set<DType> dtypes(PropagationCallGraphBuilder builder);

    /**
     * Whether the component may also be a structure, which its shapes and dtypes do not show.
     *
     * @param builder The builder.
     * @return {@code true} iff the component is a tensor on some of its datasets and a structure on
     *     others.
     */
    default boolean mayBeStructure(PropagationCallGraphBuilder builder) {
      return false;
    }
  }

  /** A component that is no tensor (⊥), such as a structure of tensors read whole. */
  private static final Component NOT_A_TENSOR =
      new Component() {
        @Override
        public Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder b) {
          return Collections.emptySet();
        }

        @Override
        public Set<DType> dtypes(PropagationCallGraphBuilder b) {
          return EnumSet.noneOf(DType.class);
        }
      };

  /**
   * The generators of the datasets a slot is filled from.
   *
   * @param builder The builder.
   * @param slot The slot's variable, possibly {@code null}.
   * @return The creators' generators, unwrapped to the datasets', or {@code null} unless the walk
   *     reaches a creator other than the slot itself and each is a dataset yielding a tuple
   *     element.
   */
  private static List<TensorGenerator> datasetCreators(
      PropagationCallGraphBuilder builder, PointsToSetVariable slot) {
    if (slot == null) return null;
    Set<PointsToSetVariable> creators = TensorGeneratorFactory.findCreators(slot, builder);
    if (creators.isEmpty() || creators.contains(slot)) return null;
    List<TensorGenerator> ret = new ArrayList<>();
    for (PointsToSetVariable creator : creators) {
      TensorGenerator part;
      try {
        part = TensorGeneratorFactory.getGenerator(creator, builder);
      } catch (IllegalArgumentException e) {
        return null;
      }
      while (part instanceof DelegatingTensorGenerator dtg) {
        TensorGenerator next = dtg.getUnderlying();
        if (next == null || next == part) break;
        part = next;
      }
      if (!(part instanceof TupleElementProvider tep && tep.yieldsTuple(builder))) return null;
      ret.add(part);
    }
    return ret;
  }

  /** How deep {@link #spreadOverReceivers} follows a chain of transformations. */
  private static final int SPREAD_DEPTH = 16;

  /**
   * The datasets the given ones are, with each transformation whose receiver may be several
   * datasets applied to each of them on its own ({@link DatasetGenerator#overReceiver}), through a
   * chain of such transformations, so that {@code (a if flag else b).shuffle(4).batch(2)} is the
   * batch of the shuffle of {@code a} and that of {@code b}: the join of both, never the first
   * (wala/ML#1009).
   *
   * @param builder The builder.
   * @param generators The datasets' generators.
   * @return The spread datasets, each yielding a tuple element, or {@code null} when none of the
   *     given ones is several, or when a spread one yields no tuple element or a chain is cyclic or
   *     deeper than {@link #SPREAD_DEPTH}, which the given reading keeps.
   */
  private static List<TensorGenerator> spreadOverReceivers(
      PropagationCallGraphBuilder builder, List<TensorGenerator> generators) {
    List<TensorGenerator> ret = new ArrayList<>();
    boolean spread = false;
    for (TensorGenerator generator : generators) {
      List<TensorGenerator> parts = spread(builder, generator, new ArrayList<>());
      if (parts == null) return null;
      if (parts.size() != 1 || parts.get(0) != generator) spread = true;
      for (TensorGenerator part : parts)
        if (!(part instanceof TupleElementProvider tep && tep.yieldsTuple(builder))) return null;
        else if (!ret.contains(part)) ret.add(part);
    }
    return spread && ret.size() > 1 ? ret : null;
  }

  /**
   * One dataset spread over its receivers, as {@link #spreadOverReceivers} does.
   *
   * @param path The transformations the chain has passed through to reach the given one.
   * @return The datasets, the given one alone when it is not spread, or {@code null} when the chain
   *     reaches a transformation again or is deeper than {@link #SPREAD_DEPTH}.
   */
  private static List<TensorGenerator> spread(
      PropagationCallGraphBuilder builder, TensorGenerator generator, List<TensorGenerator> path) {
    if (!(generator instanceof DatasetGenerator dg && dg.elementFollowsReceiver()))
      return List.of(generator);
    if (path.contains(generator) || path.size() >= SPREAD_DEPTH) return null;
    List<TensorGenerator> receivers = dg.getReceiverGenerators(builder);
    if (receivers == null || receivers.isEmpty()) return List.of(generator);
    path.add(generator);
    List<TensorGenerator> spreadReceivers = new ArrayList<>();
    for (TensorGenerator receiver : receivers) {
      while (receiver instanceof DelegatingTensorGenerator dtg) {
        TensorGenerator next = dtg.getUnderlying();
        if (next == null || next == receiver) break;
        receiver = next;
      }
      List<TensorGenerator> parts = spread(builder, receiver, path);
      if (parts == null) return null; // Abandons the whole spread, so the path is not reused.
      for (TensorGenerator part : parts)
        if (!spreadReceivers.contains(part)) spreadReceivers.add(part);
    }
    path.remove(path.size() - 1);
    if (spreadReceivers.size() <= 1) return List.of(generator);
    List<TensorGenerator> ret = new ArrayList<>();
    for (TensorGenerator receiver : spreadReceivers) ret.add(dg.overReceiver(receiver));
    return ret;
  }

  /**
   * The union of components: unknown when one is unknown and no other contributes.
   *
   * @param components The components.
   * @param structure Whether some of the joined datasets have a structure there and others a
   *     tensor.
   * @return The union.
   */
  private static Component union(List<Component> components, boolean structure) {
    return new Component() {
      @Override
      public boolean mayBeStructure(PropagationCallGraphBuilder b) {
        return structure;
      }

      @Override
      public Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder b) {
        Set<List<Dimension<?>>> ret = HashSetFactory.make();
        boolean unknown = false;
        for (Component component : components) {
          Set<List<Dimension<?>>> shapes = component.shapes(b);
          if (shapes == null) unknown = true;
          else ret.addAll(shapes);
        }
        return unknown && ret.isEmpty() ? null : ret;
      }

      @Override
      public Set<DType> dtypes(PropagationCallGraphBuilder b) {
        Set<DType> ret = EnumSet.noneOf(DType.class);
        for (Component component : components) {
          Set<DType> dtypes = component.dtypes(b);
          if (dtypes != null) ret.addAll(dtypes);
        }
        return ret;
      }
    };
  }

  /** A provider's component at the path. */
  private Component ofProvider(TupleElementProvider provider, List<Object> path) {
    return new Component() {
      @Override
      public Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder b) {
        return path.size() == 1
            ? provider.getShapesForIndex(b, (Integer) path.get(0))
            : provider.getShapesForPath(b, path);
      }

      @Override
      public Set<DType> dtypes(PropagationCallGraphBuilder b) {
        // Another pack's tensor dtypes, without its marker of a component that may be a structure,
        // which this component carries instead.
        if (path.size() == 1 && provider instanceof FitDataGenerator pack)
          return pack.tensorDTypesForIndex(b, (Integer) path.get(0));
        return path.size() == 1
            ? provider.getDTypesForIndex(b, (Integer) path.get(0))
            : provider.getDTypesForPath(b, path);
      }

      // Another pack's component, as an unpack of the step's data reads it, may be a structure
      // where that pack's is.
      @Override
      public boolean mayBeStructure(PropagationCallGraphBuilder b) {
        return path.size() == 1
            && provider instanceof FitDataGenerator pack
            && pack.mayBeStructure(b, (Integer) path.get(0));
      }
    };
  }

  /**
   * A dataset's element component, read through the dataset generator the factory resolves for the
   * slot's variable, so a transformation chain resolves as it does for user code: the element's
   * tuple component at the index when the element is a tuple, the whole element at index 0
   * otherwise, and no tensor at any other index. A longer path selects within the element's
   * component, and reaches it only when the element's structure has it.
   *
   * @return The component, or {@code null} for a longer path the element does not resolve.
   */
  private Component ofDataset(
      PropagationCallGraphBuilder builder, PointsToSetVariable slot, List<Object> path) {
    int index = (Integer) path.get(0);
    Dataset dataset = dataset(builder, slot);
    TensorGenerator generator = dataset.generator();
    List<TensorGenerator> parts = dataset.parts();
    if (generator instanceof TupleElementProvider tep && tep.yieldsTuple(builder)) {
      if (path.size() > 1 && !tep.resolvesPath(builder, path)) return null;
      // A component that is itself a structure, as `(x, mask)` in a `((x, mask), y)` element, is
      // handed to the step as that tuple or dict, which is no tensor: its members are read at the
      // longer paths, and a provider's reading of the whole would join their types into one
      // tensor of several shapes, crossed with their dtypes.
      if (tep.isStructureAt(builder, path)) return NOT_A_TENSOR;
      // Each joined dataset whose component is a structure contributes no tensor, so datasets that
      // all have a structure there make the component none. Datasets that disagree make it the
      // tensors where it is one, and also possibly a structure, which the read marks.
      if (parts != null) {
        List<Component> components = new ArrayList<>();
        boolean structure = false;
        boolean tensor = false;
        for (TensorGenerator part : parts) {
          TupleElementProvider provider = (TupleElementProvider) part;
          if (provider.isStructureAt(builder, path)) {
            structure = true;
            components.add(NOT_A_TENSOR);
          } else {
            tensor = true;
            components.add(this.ofProvider(provider, path));
          }
        }
        return union(components, structure && tensor);
      }
      return this.ofProvider(tep, path);
    }
    if (path.size() > 1) return null;
    TensorGenerator element = generator;
    return new Component() {
      @Override
      public Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder b) {
        if (index != 0 || element == null) return Collections.emptySet();
        return element.getShapes(b);
      }

      @Override
      public Set<DType> dtypes(PropagationCallGraphBuilder b) {
        if (index != 0 || element == null) return EnumSet.noneOf(DType.class);
        return element.getDTypes(b);
      }
    };
  }

  /**
   * The dataset a slot's variable holds, read as {@link #ofDataset} reads it.
   *
   * @param generator The dataset's generator, or a join of the slot's datasets.
   * @param parts The slot's datasets when there are several, or {@code null}.
   */
  private record Dataset(TensorGenerator generator, List<TensorGenerator> parts) {}

  /**
   * The dataset a slot's variable holds: the generator the factory resolves for it, unwrapped to
   * the dataset's own, or the slot's creators read in their own frame.
   *
   * @param builder The builder.
   * @param slot The slot's variable, possibly {@code null}.
   * @return The dataset.
   */
  private static Dataset dataset(PropagationCallGraphBuilder builder, PointsToSetVariable slot) {
    TensorGenerator generator = null;
    if (slot != null)
      try {
        generator = TensorGeneratorFactory.getGenerator(slot, builder);
      } catch (IllegalArgumentException e) {
        generator = null;
      }
    // A dataset element read in a loop resolves to an element generator delegating to its dataset;
    // unwrap to the dataset's own generator, as the factory does before its tuple-element dispatch.
    // A pass-through transformation's generator (`shuffle`, `prefetch`, ...) needs no unwrapping:
    // it
    // forwards every provider read to its receiver itself.
    while (generator instanceof DelegatingTensorGenerator dtg) {
      TensorGenerator next = dtg.getUnderlying();
      if (next == null || next == generator) break;
      generator = next;
    }
    // The slot's creators, the datasets the user's code may have passed, read the slot in its own
    // frame: a `fit` whose dataset is chosen as `a if flag else b` has, in a frame where the choice
    // is known, the chosen dataset as its one creator, which the factory's generator for the slot
    // may not be. Where the choice is either, the slot has a creator per dataset, and its element
    // is their join.
    List<TensorGenerator> parts = datasetCreators(builder, slot);
    // A transformation of a dataset that is itself either of several, as `(a if flag else
    // b).batch(2)`, is the slot's one creator, and its reads forward to one of its receiver's
    // datasets; applied to each of them on its own, it is a part of the join like a creator.
    List<TensorGenerator> spread =
        spreadOverReceivers(
            builder, parts != null ? parts : generator == null ? List.of() : List.of(generator));
    if (spread != null) parts = spread;
    if (parts != null && parts.size() == 1) {
      generator = parts.get(0);
      parts = null;
    } else if (parts != null) generator = new DatasetCreatorJoin(slot, parts);
    return new Dataset(generator, parts);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A dataset fed to the step declares the structure of its element: the selector picks
   *     the element's component, the inputs at {@code 0}, and the rest of the path is the
   *     dataset's. Datasets that the slot may hold either of must declare one length. A tensor fed
   *     to the step declares none.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.isEmpty() || !(path.get(0) instanceof Integer)) return OptionalInt.empty();
    for (InstanceKey x : this.slot(builder, 0)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(x);
      if (asin == null || !isDataset(asin.concreteType().getReference()))
        return OptionalInt.empty();
    }
    Dataset dataset = dataset(builder, this.slotVariable(builder, this.allocation, 0));
    List<TensorGenerator> providers =
        dataset.parts() != null ? dataset.parts() : Collections.singletonList(dataset.generator());
    int ret = -1;
    for (TensorGenerator provider : providers) {
      if (!(provider instanceof TupleElementProvider tep) || !tep.yieldsTuple(builder))
        return OptionalInt.empty();
      OptionalInt arity = tep.declaredArity(builder, path);
      if (arity.isEmpty() || (ret >= 0 && ret != arity.getAsInt())) return OptionalInt.empty();
      ret = arity.getAsInt();
    }
    return ret < 0 ? OptionalInt.empty() : OptionalInt.of(ret);
  }

  /** The values' own types. */
  private Component ofValues(OrdinalSet<InstanceKey> values) {
    return new Component() {
      @Override
      public Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder b) {
        if (values == null || values.isEmpty()) return Collections.emptySet();
        return FitDataGenerator.this.getShapesOfValue(b, values);
      }

      @Override
      public Set<DType> dtypes(PropagationCallGraphBuilder b) {
        if (values == null || values.isEmpty()) return EnumSet.noneOf(DType.class);
        return FitDataGenerator.this.getDTypesOfValue(b, values);
      }
    };
  }
}
