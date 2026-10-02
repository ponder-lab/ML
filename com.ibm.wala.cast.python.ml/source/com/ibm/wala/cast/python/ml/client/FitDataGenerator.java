package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATA_PACKAGE_PREFIX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIT_DATA_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNPACKED_DATA_TYPE;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
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
    for (Component component : this.components(builder, index)) {
      Set<List<Dimension<?>>> shapes = component.shapes(builder);
      if (shapes == null) unknown = true;
      else ret.addAll(shapes);
    }
    return unknown && ret.isEmpty() ? null : ret;
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Component component : this.components(builder, index)) {
      Set<DType> dtypes = component.dtypes(builder);
      if (dtypes != null) ret.addAll(dtypes);
    }
    return ret;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dtypes = this.getDTypesForIndex(builder, index);
    Set<TensorType> ret = HashSetFactory.make();
    if (shapes == null) {
      for (DType dtype : dtypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), null));
      return ret;
    }
    for (List<Dimension<?>> shape : shapes)
      for (DType dtype : dtypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), shape));
    return ret;
  }

  /** The values a read of the given index resolves to, each with the way it is read. */
  private List<Component> components(PropagationCallGraphBuilder builder, int index) {
    List<Component> ret = new ArrayList<>();
    if (this.unpack) {
      // A dataset element read in a loop is the dataset object in the pointer analysis, whose set
      // also carries the object's field values (its method objects, its source tensors); the
      // dataset reading covers the element, and those members are not components of it.
      for (InstanceKey data : this.slot(builder, 0)) {
        AllocationSiteInNode asin = getAllocationSiteInNode(data);
        if (asin != null && isDataset(asin.concreteType().getReference())) {
          ret.add(this.ofDataset(builder, this.slotVariable(builder, this.allocation, 0), index));
          return ret;
        }
      }
      for (InstanceKey data : this.slot(builder, 0)) {
        AllocationSiteInNode asin = getAllocationSiteInNode(data);
        TypeReference type = asin == null ? null : asin.concreteType().getReference();
        if (asin != null && describes(type))
          ret.add(this.ofProvider(new FitDataGenerator(asin.getNode(), asin), index));
        else if (asin != null && PythonTypes.tuple.equals(type))
          ret.add(this.ofValues(this.field(builder, asin, index)));
        else if (index == 0) ret.add(this.ofValues(this.singleton(builder, data)));
      }
      return ret;
    }
    for (InstanceKey x : this.slot(builder, 0)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(x);
      if (asin != null && isDataset(asin.concreteType().getReference())) {
        ret.add(this.ofDataset(builder, this.slotVariable(builder, this.allocation, 0), index));
        return ret;
      }
    }
    ret.add(this.ofValues(this.slot(builder, index)));
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
        builder.getPointerAnalysis().getPointsToSet(builder.getPointerKeyForInstanceField(asin, f));
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
  }

  /** A provider's component at the index. */
  private Component ofProvider(TupleElementProvider provider, int index) {
    return new Component() {
      @Override
      public Set<List<Dimension<?>>> shapes(PropagationCallGraphBuilder b) {
        return provider.getShapesForIndex(b, index);
      }

      @Override
      public Set<DType> dtypes(PropagationCallGraphBuilder b) {
        return provider.getDTypesForIndex(b, index);
      }
    };
  }

  /**
   * A dataset's element component, read through the dataset generator the factory resolves for the
   * slot's variable, so a transformation chain resolves as it does for user code: the element's
   * tuple component at the index when the element is a tuple, the whole element at index 0
   * otherwise, and no tensor at any other index.
   */
  private Component ofDataset(
      PropagationCallGraphBuilder builder, PointsToSetVariable slot, int index) {
    TensorGenerator generator = null;
    if (slot != null)
      try {
        generator = TensorGeneratorFactory.getGenerator(slot, builder);
      } catch (IllegalArgumentException e) {
        generator = null;
      }
    // A dataset element read in a loop resolves to an element generator delegating to its dataset,
    // and a pass-through transformation to its receiver; unwrap to the provider, as the factory
    // does before its own tuple-element dispatch.
    boolean changed = true;
    while (changed && generator != null) {
      changed = false;
      if (generator instanceof DelegatingTensorGenerator dtg) {
        TensorGenerator next = dtg.getUnderlying();
        if (next != null && next != generator) {
          generator = next;
          changed = true;
        }
      }
      if (!changed && generator.getClass() == DatasetGenerator.class) {
        TensorGenerator receiver = ((DatasetGenerator) generator).getReceiverGenerator(builder);
        if (receiver != null && receiver != generator) {
          generator = receiver;
          changed = true;
        }
      }
    }
    if (generator instanceof TupleElementProvider tep && tep.yieldsTuple(builder))
      return this.ofProvider(tep, index);
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
